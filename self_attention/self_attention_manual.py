#基于pytorch的self-attention
import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,dim_input, dim_q, dim_v):
        '''
        参数说明：
        dim_input: 输入数据x中每一个样本的向量维度
        dim_q:     Q矩阵的列向维度, 在运算时dim_q要和dim_k保持一致;
                   因为需要进行: K^T*Q运算, 结果为：[dim_input, dim_input]方阵
        dim_v:     V矩阵的列项维度,
        '''
        super().__init__()
        
        self.dim_input = dim_input
        self.dim_q = dim_q
        self.dim_k = dim_q
        self.dim_v = dim_v
        self._norm_fact = 1 / math.sqrt(self.dim_k)
        self.linear_q = nn.Linear(self.dim_input, self.dim_q, bias=False)
        self.linear_k = nn.Linear(self.dim_input, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_input, self.dim_v, bias=False)
        
    def forward(self, x_in):
        '''
        输入：
            x_in: [batch_size, seq_len, dim_input]
        输出：
            out: [batch_size, seq_len, dim_v]
        '''
        # 计算Q矩阵
        q_linear = self.linear_q(x_in)
        # 计算K矩阵
        k_linear = self.linear_k(x_in)
        # 计算V矩阵
        v_linear = self.linear_v(x_in)
        print(f'x_in.shape:{x_in.shape} \nQ.shape:{q_linear.shape} \nK.shape: {k_linear.shape} \nV.shape:{v_linear.shape}')
        # 计算Q*K^T
        a = torch.bmm(q_linear, k_linear.transpose(1, 2)) * self._norm_fact
        # softmax处理
        a_p = torch.softmax(a, dim=-1)
        print(f'a_p.shape:{a_p.shape}')
        # 获得输出
        out = torch.bmm(a_p, v_linear)
        print(f'out.shape:{out.shape}')
        return out
    
if __name__ == "__main__":
    
    batch_size = 5
    seq_len = 10
    dim_input = 20
    dim_q = 15
    dim_v = 25
    attention = SelfAttention(dim_input, dim_q, dim_v)
    
    x_in = torch.randn(batch_size, seq_len, dim_input)
    out = attention(x_in)
    print(out)