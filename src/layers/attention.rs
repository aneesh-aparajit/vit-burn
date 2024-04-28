use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

#[derive(Debug, Module)]
pub struct SelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dropout: Dropout,
}

#[derive(Debug, Config)]
pub struct SelfAttentionConfig {
    head_dim: usize,
    embedding_dim: usize,
    dropout: f64,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        SelfAttention {
            query: LinearConfig::new(self.embedding_dim, self.head_dim).init(device),
            key: LinearConfig::new(self.embedding_dim, self.head_dim).init(device),
            value: LinearConfig::new(self.embedding_dim, self.head_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> SelfAttention<B> {
    pub fn forward(self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let q: Tensor<B, 3> = self.query.forward(inputs.clone());
        let k: Tensor<B, 3> = self.key.forward(inputs.clone());
        let v: Tensor<B, 3> = self.value.forward(inputs.clone());

        let [_, _, head_dim] = q.dims();

        let mut attention = softmax(
            q.clone().matmul(k.permute([0, 2, 1])) / (head_dim as f64).sqrt(),
            2,
        );
        attention = attention.matmul(v);
        return attention;
    }
}

