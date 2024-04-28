use super::attention::{SelfAttention, SelfAttentionConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Debug, Module)]
pub struct MultiHeadAttention<B: Backend> {
    self_attn: Vec<SelfAttention<B>>,
    output: Linear<B>,
    dropout: Dropout,
}

#[derive(Debug, Config)]
pub struct MultiHeadAttentionConfig {
    num_heads: usize,
    embedding_dim: usize,
    head_dim: usize,
    #[config(default = "0.0")]
    dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let mut attns: Vec<SelfAttention<B>> = vec![];

        for _ in 0..self.num_heads {
            attns.push(
                SelfAttentionConfig::new(self.head_dim, self.embedding_dim, self.dropout)
                    .init(device),
            );
        }

        MultiHeadAttention {
            self_attn: attns,
            output: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut head_outputs = vec![];
        for i in 0..self.self_attn.len() {
            head_outputs.push(self.self_attn[i].clone().forward(inputs.clone()));
        }
        let output = Tensor::cat(head_outputs, 2);
        let output = self.output.forward(output);
        self.dropout.forward(output)
    }
}

