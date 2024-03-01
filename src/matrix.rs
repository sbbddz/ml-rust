use core::fmt;

use crate::{get_random_float, sigmoid};

pub struct Matrix {
    rows: usize,
    cols: usize,
    imp: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let mat = Matrix {
            rows,
            cols,
            imp: vec![0f32; cols * rows],
        };
        mat
    }

    pub fn new_rand(rows: usize, cols: usize) -> Matrix {
        let mat = Matrix {
            rows,
            cols,
            imp: vec![get_random_float(); cols * rows],
        };
        mat
    }

    pub fn from_literal(vec: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        let mat = Matrix {
            rows,
            cols,
            imp: vec,
        };
        mat
    }

    pub fn dot(&self, matrix: &Matrix) -> Matrix {
        if self.cols != matrix.rows {
            panic!("You cannot multiply a matrix which its cols does not match the rows of the other matrix");
        }

        let mut new_matrix = Matrix::new(self.rows, matrix.cols);

        let n = self.cols;

        for i in 0..new_matrix.rows {
            for j in 0..new_matrix.cols {
                for k in 0..n {
                    *new_matrix.at_mut(i, j).unwrap() +=
                        self.at(i, k).unwrap() * matrix.at(k, j).unwrap();
                }
            }
        }
        new_matrix
    }

    pub fn sum(&mut self, matrix: &Matrix) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.at_mut(i, j).unwrap() += matrix.at(i, j).unwrap();
            }
        }
    }

    pub fn at(&self, x: usize, y: usize) -> Option<&f32> {
        self.imp.get(x * self.cols + y)
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> Option<&mut f32> {
        self.imp.get_mut(x * self.cols + y)
    }

    pub fn rand(&mut self, min: f32, max: f32) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.imp[i * self.cols + j] = get_random_float() * (max - min) + min;
            }
        }
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.at(i, j).unwrap();
                *self.at_mut(i, j).unwrap() = sigmoid(*value);
            }
        }
    }

    fn sigmoid_value(x: f32) -> f32 {
        return 1f32 / (1f32 + x.exp());
    }

    pub fn fill(&mut self, value: f32) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.imp[i * self.cols + j] = value;
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = format!("{:?}", self.at(i, j));
                string.push_str(&value);
                string.push_str("  ");
            }
            string.push_str("\n");
        }

        write!(f, "{}", string)
    }
}

struct XorModel {
    // INPUT
    x: Matrix,
    // LAYER 1
    weights_layer_1: Matrix,
    bias_layer_1: Matrix,
    // LAYER 2
    weights_layer_2: Matrix,
    bias_layer_2: Matrix,
}

impl XorModel {
    pub fn new() -> XorModel {
        XorModel {
            x: Matrix::new(1, 2),
            weights_layer_1: Matrix::new_rand(2, 2),
            bias_layer_1: Matrix::new_rand(1, 2),
            weights_layer_2: Matrix::new_rand(2, 1),
            bias_layer_2: Matrix::new_rand(1, 1),
        }
    }

    pub fn forward(&mut self) -> Option<f32> {
        let mut activation_matrix = self.x.dot(&self.weights_layer_1);
        activation_matrix.sum(&self.bias_layer_1);
        activation_matrix.sigmoid();
        let mut activation_matrix = activation_matrix.dot(&self.weights_layer_2);
        activation_matrix.sum(&self.bias_layer_2);
        activation_matrix.sigmoid();
        activation_matrix.at(0, 0).copied()
    }

    pub fn set_input(&mut self, x: f32, y: f32) {
        *self.x.at_mut(0, 0).unwrap() = x;
        *self.x.at_mut(0, 1).unwrap() = y;
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, XorModel};

    #[test]
    fn xor() {
        let mut xor_model = XorModel::new();
        for i in 0..2 {
            for j in 0..2 {
                xor_model.set_input(i as f32, j as f32);
                let value = xor_model.forward();
                println!("{} | {} = {:?}", i, j, value);
            }
        }
        println!();
    }

    #[test]
    fn exploring_xor() {
        // let mut weights_layer_1 = Matrix::new(2, 2);
        // weights_layer_1.rand(0f32, 1f32);
        // println!("{}", weights_layer_1);
        // let mut bias_layer_1 = Matrix::new(1, 2);
        // bias_layer_1.rand(0f32, 1f32);
        // println!("{}", bias_layer_1);

        // let mut weights_layer_2 = Matrix::new(2, 1);
        // weights_layer_2.rand(0f32, 1f32);
        // println!("{}", weights_layer_2);

        // let mut bias_layer_2 = Matrix::new(1, 1);
        // bias_layer_2.rand(0f32, 1f32);
        // println!("{}", bias_layer_2);

        // let mut input_x = Matrix::new(1, 2);
        // *input_x.at_mut(0, 0).unwrap() = 0f32;
        // *input_x.at_mut(0, 1).unwrap() = 1f32;

        // // FORWARD
        // let mut activation_matrix = input_x.dot(&weights_layer_1);
        // activation_matrix.sum(&bias_layer_1);
        // activation_matrix.sigmoid();
        // println!("activation matrix: {}", activation_matrix);
    }
}
