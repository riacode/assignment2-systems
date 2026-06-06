from __future__ import annotations

import math
from collections import namedtuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    triton = namedtuple("triton", ["jit", "make_block_ptr", "testing"])
    triton.jit = lambda x: x
    tl = namedtuple("tl", ["constexpr"])
    _HAS_TRITON = False


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)
    key_limit = N_KEYS
    if is_causal:
        key_limit = tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE)

    for _ in range(tl.cdiv(key_limit, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        valid_mask = (query_offsets[:, None] < N_QUERIES) & (key_offsets[None, :] < N_KEYS)
        S_i_j = tl.where(valid_mask, S_i_j, -1.0e6)
        if is_causal:
            causal_mask = query_offsets[:, None] >= key_offsets[None, :]
            S_i_j += tl.where(causal_mask, 0.0, -1.0e6)

        m_i_j = tl.maximum(m_i, tl.max(S_i_j, axis=1))
        P_i_j = tl.exp(S_i_j - m_i_j[:, None])
        P_i_j = tl.where(valid_mask, P_i_j, 0.0)

        old_scale = tl.exp(m_i - m_i_j)
        l_i_j = old_scale * l_i + tl.sum(P_i_j, axis=1)
        O_i = old_scale[:, None] * O_i + tl.dot(P_i_j.to(V_j.dtype), V_j)

        m_i = m_i_j
        l_i = l_i_j
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        key_offsets += K_TILE_SIZE

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    key_offsets = tl.arange(0, K_TILE_SIZE)
    dim_offsets = tl.arange(0, D)
    Q_i = tl.load(
        Q_ptr + batch_index * stride_qb + query_offsets[:, None] * stride_qq + dim_offsets[None, :] * stride_qd,
        mask=(query_offsets[:, None] < N_QUERIES),
        other=0.0,
    )
    O_i = tl.load(
        O_ptr + batch_index * stride_ob + query_offsets[:, None] * stride_oq + dim_offsets[None, :] * stride_od,
        mask=(query_offsets[:, None] < N_QUERIES),
        other=0.0,
    )
    dO_i = tl.load(
        dO_ptr + batch_index * stride_dob + query_offsets[:, None] * stride_doq + dim_offsets[None, :] * stride_dod,
        mask=(query_offsets[:, None] < N_QUERIES),
        other=0.0,
    )
    L_i = tl.load(
        L_ptr + batch_index * stride_lb + query_offsets * stride_lq,
        mask=query_offsets < N_QUERIES,
        other=0.0,
    )
    D_i = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    key_limit = N_KEYS
    if is_causal:
        key_limit = tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE)

    for _ in range(tl.cdiv(key_limit, K_TILE_SIZE)):
        K_j = tl.load(
            K_ptr + batch_index * stride_kb + key_offsets[:, None] * stride_kk + dim_offsets[None, :] * stride_kd,
            mask=(key_offsets[:, None] < N_KEYS),
            other=0.0,
        )
        V_j = tl.load(
            V_ptr + batch_index * stride_vb + key_offsets[:, None] * stride_vk + dim_offsets[None, :] * stride_vd,
            mask=(key_offsets[:, None] < N_KEYS),
            other=0.0,
        )

        S = tl.dot(Q_i, tl.trans(K_j)) * scale
        mask = (query_offsets[:, None] < N_QUERIES) & (key_offsets[None, :] < N_KEYS)
        if is_causal:
            mask = mask & (query_offsets[:, None] >= key_offsets[None, :])
        P = tl.exp(S - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        dP = tl.dot(dO_i, tl.trans(V_j))
        dS = P * (dP - D_i[:, None])
        dQ_i += tl.dot(dS.to(K_j.dtype), K_j) * scale
        key_offsets += K_TILE_SIZE

    tl.store(
        dQ_ptr + batch_index * stride_dqb + query_offsets[:, None] * stride_dqq + dim_offsets[None, :] * stride_dqd,
        dQ_i,
        mask=query_offsets[:, None] < N_QUERIES,
    )

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    key_start = key_tile_index * K_TILE_SIZE
    query_start = 0
    if is_causal:
        query_start = tl.minimum(N_QUERIES, key_start)
    query_offsets = query_start + tl.arange(0, Q_TILE_SIZE)
    key_offsets = key_start + tl.arange(0, K_TILE_SIZE)
    dim_offsets = tl.arange(0, D)

    K_j = tl.load(
        K_ptr + batch_index * stride_kb + key_offsets[:, None] * stride_kk + dim_offsets[None, :] * stride_kd,
        mask=(key_offsets[:, None] < N_KEYS),
        other=0.0,
    )
    V_j = tl.load(
        V_ptr + batch_index * stride_vb + key_offsets[:, None] * stride_vk + dim_offsets[None, :] * stride_vd,
        mask=(key_offsets[:, None] < N_KEYS),
        other=0.0,
    )
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for _ in range(tl.cdiv(N_QUERIES - query_start, Q_TILE_SIZE)):
        Q_i = tl.load(
            Q_ptr + batch_index * stride_qb + query_offsets[:, None] * stride_qq + dim_offsets[None, :] * stride_qd,
            mask=(query_offsets[:, None] < N_QUERIES),
            other=0.0,
        )
        O_i = tl.load(
            O_ptr + batch_index * stride_ob + query_offsets[:, None] * stride_oq + dim_offsets[None, :] * stride_od,
            mask=(query_offsets[:, None] < N_QUERIES),
            other=0.0,
        )
        dO_i = tl.load(
            dO_ptr + batch_index * stride_dob + query_offsets[:, None] * stride_doq + dim_offsets[None, :] * stride_dod,
            mask=(query_offsets[:, None] < N_QUERIES),
            other=0.0,
        )
        L_i = tl.load(
            L_ptr + batch_index * stride_lb + query_offsets * stride_lq,
            mask=query_offsets < N_QUERIES,
            other=0.0,
        )

        S = tl.dot(Q_i, tl.trans(K_j)) * scale
        mask = (query_offsets[:, None] < N_QUERIES) & (key_offsets[None, :] < N_KEYS)
        if is_causal:
            mask = mask & (query_offsets[:, None] >= key_offsets[None, :])
        P = tl.exp(S - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        D_i = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
        dV_j += tl.dot(tl.trans(P).to(dO_i.dtype), dO_i)
        dP = tl.dot(dO_i, tl.trans(V_j))
        dS = P * (dP - D_i[:, None])
        dK_j += tl.dot(tl.trans(dS).to(Q_i.dtype), Q_i) * scale
        query_offsets += Q_TILE_SIZE

    tl.store(
        dK_ptr + batch_index * stride_dkb + key_offsets[:, None] * stride_dkk + dim_offsets[None, :] * stride_dkd,
        dK_j,
        mask=key_offsets[:, None] < N_KEYS,
    )
    tl.store(
        dV_ptr + batch_index * stride_dvb + key_offsets[:, None] * stride_dvk + dim_offsets[None, :] * stride_dvd,
        dV_j,
        mask=key_offsets[:, None] < N_KEYS,
    )


def flash_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)
    Q = Q.float()
    K = K.float()
    V = V.float()
    O = O.float()
    dO = dO.float()
    L = L.float()

    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if is_causal:
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        q_idx = torch.arange(N_q, device=Q.device)
        k_idx = torch.arange(N_k, device=Q.device)
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        S = torch.where(causal_mask, S, -1.0e6)

    P = torch.exp(S - L[..., :, None])
    D_vec = torch.sum(O * dO, dim=-1)
    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    dS = P * (dP - D_vec[..., :, None])
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-1, -2), Q) * scale
    return dQ, dK, dV


flash_backward_pytorch_compiled = torch.compile(flash_backward_pytorch, backend="aot_eager")


def flash_backward_triton(L, Q, K, V, O, dO, is_causal):
    *batch_dims, N_q, d = Q.shape
    N_k = K.shape[-2]
    batch_size = math.prod(batch_dims) if batch_dims else 1

    Q = Q.reshape(batch_size, N_q, d)
    K = K.reshape(batch_size, N_k, d)
    V = V.reshape(batch_size, N_k, d)
    O = O.reshape(batch_size, N_q, d)
    dO = dO.reshape(batch_size, N_q, d)
    L = L.reshape(batch_size, N_q)

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    B_q = 32
    B_k = 64
    scale = 1 / math.sqrt(d)

    flash_bwd_dq_kernel[(math.ceil(N_q / B_q), batch_size)](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        N_QUERIES=N_q, N_KEYS=N_k,
        scale=scale,
        D=d,
        Q_TILE_SIZE=B_q,
        K_TILE_SIZE=B_k,
        is_causal=is_causal,
    )
    flash_bwd_dkdv_kernel[(math.ceil(N_k / B_k), batch_size)](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        N_QUERIES=N_q, N_KEYS=N_k,
        scale=scale,
        D=d,
        Q_TILE_SIZE=B_q,
        K_TILE_SIZE=B_k,
        is_causal=is_causal,
    )
    return (
        dQ.reshape(*batch_dims, N_q, d),
        dK.reshape(*batch_dims, N_k, d),
        dV.reshape(*batch_dims, N_k, d),
    )


class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *batch_dims, N_q, d = Q.shape
        N_k = K.shape[-2]
        batch_size = math.prod(batch_dims) if batch_dims else 1
        scale = 1 / math.sqrt(d)

        Q = Q.reshape(batch_size, N_q, d)
        K = K.reshape(batch_size, N_k, d)
        V = V.reshape(batch_size, N_k, d)

        B_q = 16
        B_k = 64
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        O = torch.empty_like(Q)
        L = torch.empty(batch_size, N_q, device=Q.device, dtype=torch.float32)

        for i in range(T_q):
            q_start = i * B_q
            q_end = q_start + B_q
            Q_i = Q[:, q_start:q_end, :].float()
            B_q_i = Q_i.shape[-2]
            O_i = torch.zeros((batch_size, B_q_i, d), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((batch_size, B_q_i), device=Q.device, dtype=torch.float32)
            m_i = torch.full((batch_size, B_q_i), -torch.inf, device=Q.device, dtype=torch.float32)

            for j in range(T_k):
                k_start = j * B_k
                k_end = k_start + B_k
                K_j = K[:, k_start:k_end, :].float()
                V_j = V[:, k_start:k_end, :].float()
                S_i_j = torch.matmul(Q_i, K_j.transpose(-1, -2)) * scale
                m_i_j = torch.maximum(m_i, S_i_j.max(dim=-1).values)
                P_i_j = torch.exp(S_i_j - m_i_j[:, :, None])
                old_scale = torch.exp(m_i - m_i_j)
                l_i_j = old_scale * l_i + P_i_j.sum(dim=-1)
                O_i = old_scale[:, :, None] * O_i + torch.matmul(P_i_j, V_j)
                m_i = m_i_j
                l_i = l_i_j

            O[:, q_start:q_start + B_q_i, :] = (O_i / l_i[:, :, None]).to(O.dtype)
            L[:, q_start:q_start + B_q_i] = m_i + torch.log(l_i)

        O = O.reshape(*batch_dims, N_q, d)
        L = L.reshape(*batch_dims, N_q)
        Q = Q.reshape(*batch_dims, N_q, d)
        K = K.reshape(*batch_dims, N_k, d)
        V = V.reshape(*batch_dims, N_k, d)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_pytorch_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is required to run FlashAttention2Triton.")

        *batch_dims, N_q, d = Q.shape
        N_k = K.shape[-2]
        batch_size = math.prod(batch_dims) if batch_dims else 1

        Q = Q.reshape(batch_size, N_q, d)
        K = K.reshape(batch_size, N_k, d)
        V = V.reshape(batch_size, N_k, d)

        B_q = 64
        B_k = 64
        T_q = math.ceil(N_q / B_q)
        scale = 1 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty(batch_size, N_q, device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k,
            scale=scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )

        O = O.reshape(*batch_dims, N_q, d)
        L = L.reshape(*batch_dims, N_q)
        Q = Q.reshape(*batch_dims, N_q, d)
        K = K.reshape(*batch_dims, N_k, d)
        V = V.reshape(*batch_dims, N_k, d)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_triton(L, Q, K, V, O, dO, ctx.is_causal)
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None
