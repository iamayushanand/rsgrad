use rsgrad_nn::loss::L2loss;
use rsgrad_primitive::tensor::Tensor;
use rsgrad_nn::layer::ReLU;
use rsgrad_nn::layer::Linear;
use rsgrad_nn::optimizer::SGD;
use std::rc::Rc;
use std::cell::RefCell;
use rand::prelude::*;

struct NeuralNet {
    layer_1: Linear,
    layer_2: Linear,
    layer_3: Linear,
    activation: ReLU
}

impl NeuralNet {
    pub fn new() -> NeuralNet {
        NeuralNet {layer_1: Linear::new(2, 6), layer_2: Linear::new(6, 2), layer_3: Linear::new(2, 1), activation: ReLU{}}
    }

    pub fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let mut res_1 = self.layer_1.forward(x);
        let mut res_1_ac = self.activation.forward(res_1);
        let mut res_2 = self.activation.forward(self.layer_2.forward(res_1_ac));
        let mut res_3 = self.layer_3.forward(res_2);
        res_3
    }

    pub fn params(&self) -> Vec<Rc<RefCell<Tensor>>> {
        let mut params: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        params.push(self.layer_1.param.clone());
        params.push(self.layer_2.param.clone());
        params.push(self.layer_3.param.clone());
        params
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let model = NeuralNet::new();
    let optim = SGD::new(model.params(),0.001);
    let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 1])));
    let loss = L2loss{};
    let mut train_data: Vec<(f32, f32)> = Vec::new();
    for n in 0..3000 {
        let a: f32 = 2.0*rng.gen::<f32>();
        let b: f32 = 2.0*rng.gen::<f32>();
        train_data.push((a, b));
    }

    let mut val_data: Vec<(f32, f32)> = Vec::new();
    for n in 0..3000 {
        let a: f32 = 2.0*rng.gen::<f32>();
        let b: f32 = 2.0*rng.gen::<f32>();
        val_data.push((a, b));
    }

    for epoch in 0..100 {
        let mut running_loss: f32 = 0.0;
        for n in 0..3000 {
            let a: f32 = train_data[n].0;
            let b: f32 = train_data[n].1;
            let c: f32 = a.exp()+b.exp();
            let mut x: Rc<RefCell<Tensor>> = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 2])));
            let mut t: Rc<RefCell<Tensor>> = Rc::new(RefCell::new(Tensor::constant_fill(c, &[1,1])));
            *x.borrow_mut().at(&[0,0]) = a;
            *x.borrow_mut().at(&[0,1]) = b;
            optim.zero_grad();
            let mut res = model.forward(x);
            let mut loss_val = loss.forward(res.clone(), t);
            running_loss += loss_val.buffer[0];
            loss_val.backward(init_grad.clone());
            optim.step();
        }
        if epoch%10==0 {
            println!("epoch:{} ==> loss: {}", epoch, running_loss/3000.0);
        }
    }

    let mut val_running_loss = 0.0;
    for n in 0..3000 {
        let a: f32 = val_data[n].0;
        let b: f32 = val_data[n].1;
        let c: f32 = a.exp()+b.exp();
        let mut x: Rc<RefCell<Tensor>> = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 2])));
        let mut t: Rc<RefCell<Tensor>> = Rc::new(RefCell::new(Tensor::constant_fill(c, &[1,1])));
        *x.borrow_mut().at(&[0,0]) = a;
        *x.borrow_mut().at(&[0,1]) = b;
        let mut res = model.forward(x);
        let mut loss_val = loss.forward(res.clone(), t);
        val_running_loss += loss_val.buffer[0];
        if n%1000==0 {
            println!("n:{} expected: {} got: {}", n, c, res.borrow().buffer[0]);
        }
    }
    println!("Val running loss: {}", val_running_loss/3000.0);
}
