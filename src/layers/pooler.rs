use burn::{config::Config, module::Module, nn::{Linear, LinearConfig}, tensor::{backend::Backend, Tensor}};


#[derive(Debug, Module)]
pub struct Pooler<B: Backend> {
    dense: Linear<B>,
}

#[derive(Debug, Config)]
pub struct PoolerConfig {
    embedding_dim: usize,
    out_features: usize,
}

impl PoolerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pooler<B> {
        Pooler { 
            dense: LinearConfig::new(self.embedding_dim, self.out_features).init(device),
        }
    }
}

impl <B: Backend> Pooler<B> {
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        self.dense.forward(inputs)
    }
}
