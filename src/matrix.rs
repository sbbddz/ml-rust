use core::fmt;

use crate::get_random_float;

#[derive(Debug)]
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
        let mut mat = Matrix {
            rows,
            cols,
            imp: vec![0f32; cols * rows],
        };
        mat.rand(0f32, 1f32);
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
        assert!(
            self.cols == matrix.rows,
            "You cannot multiply a matrix which its cols does not match the rows of the other matrix"
        );

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
        assert!(
            self.shape() == matrix.shape(),
            "You cannot sum two matrix of different shape"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.at_mut(i, j).unwrap() += matrix.at(i, j).unwrap();
            }
        }
    }

    pub fn rand(&mut self, min: f32, max: f32) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.imp[i * self.cols + j] = get_random_float() * (max - min) + min;
            }
        }
    }

    pub fn at(&self, x: usize, y: usize) -> Option<&f32> {
        self.imp.get(x * self.cols + y)
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> Option<&mut f32> {
        self.imp.get_mut(x * self.cols + y)
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.at(i, j).unwrap();
                *self.at_mut(i, j).unwrap() = Matrix::sigmoid_value(*value);
            }
        }
    }

    pub fn row(&self, row_index: usize) -> Matrix {
        let stride = self.cols;
        let row_index = row_index * stride;
        let vec = &self.imp[row_index..row_index + stride];
        Matrix {
            cols: self.cols,
            rows: 1,
            imp: vec.to_vec(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn sigmoid_value(x: f32) -> f32 {
        return 1f32 / (1f32 + x.exp());
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

#[derive(Debug)]
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

    pub fn forward(&self) -> Matrix {
        let mut activation_matrix = self.x.dot(&self.weights_layer_1);
        activation_matrix.sum(&self.bias_layer_1);
        activation_matrix.sigmoid();
        let mut activation_matrix = activation_matrix.dot(&self.weights_layer_2);
        activation_matrix.sum(&self.bias_layer_2);
        activation_matrix.sigmoid();
        activation_matrix
    }

    pub fn set_input_matrix(&mut self, input_matrix: Matrix) {
        assert!(input_matrix.shape() == self.x.shape());
        self.x = input_matrix
    }

    pub fn loss(&mut self, training_input: &Matrix, training_output: &Matrix) -> f32 {
        assert!(training_input.rows == training_output.rows);

        let mut result = 0f32;

        for i in 0..training_input.rows {
            let row = training_input.row(i);
            self.set_input_matrix(row);
            let f_result = self.forward();
            let expected_matrix = training_output.row(i);

            assert!(f_result.shape() == expected_matrix.shape());

            // Computing difference like this allows us to compute the diff of two matrix maybe, if
            // in the future input and output should not be a value but a matrix
            for j in 0..f_result.cols {
                let diff = f_result.at(0, j).unwrap() - expected_matrix.at(0, j).unwrap();
                result += diff * diff;
            }
        }

        result / training_input.rows as f32
    }

    pub fn train(&mut self, training_input: &Matrix, training_output: &Matrix) -> () {
        let loss = self.loss(training_input, training_output);
        println!("loss={loss}");

        for _ in 0..100*10000 {
            let diff = self.finite_diff(training_input, training_output, 0.1);
            self.apply_diff(&diff, 0.1);
            let loss = self.loss(training_input, training_output);
            println!("loss={loss}");
        }

    }

    pub fn apply_diff(&mut self, diff: &XorModel, learning_rate: f32) -> () {
        for i in 0..self.weights_layer_1.rows {
            for j in 0..self.weights_layer_1.cols {
                let actual_diff = *diff.weights_layer_1.at(i, j).unwrap();
                *self.weights_layer_1.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
            }
        }

        for i in 0..self.bias_layer_1.rows {
            for j in 0..self.bias_layer_1.cols {
                let actual_diff = *diff.bias_layer_1.at(i, j).unwrap();
                *self.bias_layer_1.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
            }
        }

        for i in 0..self.weights_layer_2.rows {
            for j in 0..self.weights_layer_2.cols {
                let actual_diff = *diff.weights_layer_2.at(i, j).unwrap();
                *self.weights_layer_2.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
            }
        }

        for i in 0..self.bias_layer_2.rows {
            for j in 0..self.bias_layer_2.cols {
                let actual_diff = *diff.bias_layer_2.at(i, j).unwrap();
                *self.bias_layer_2.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
            }
        }
    }

    pub fn finite_diff(
        &mut self,
        training_input: &Matrix,
        training_output: &Matrix,
        epsilon: f32,
    ) -> XorModel {
        let mut gradient = XorModel::new();
        let initial_loss = self.loss(training_input, training_output);

        for i in 0..self.weights_layer_1.rows {
            for j in 0..self.weights_layer_1.cols {
                let actual = *self.weights_layer_1.at(i, j).unwrap();
                *self.weights_layer_1.at_mut(i, j).unwrap() = actual + epsilon;
                *gradient.weights_layer_1.at_mut(i, j).unwrap() =
                    (self.loss(training_input, training_output) - initial_loss) / epsilon;
                *self.weights_layer_1.at_mut(i, j).unwrap() = actual;
            }
        }

        for i in 0..self.bias_layer_1.rows {
            for j in 0..self.bias_layer_1.cols {
                let actual = *self.bias_layer_1.at(i, j).unwrap();
                *self.bias_layer_1.at_mut(i, j).unwrap() = actual + epsilon;
                *gradient.bias_layer_1.at_mut(i, j).unwrap() =
                    (self.loss(training_input, training_output) - initial_loss) / epsilon;
                *self.bias_layer_1.at_mut(i, j).unwrap() = actual;
            }
        }

        for i in 0..self.weights_layer_2.rows {
            for j in 0..self.weights_layer_2.cols {
                let actual = *self.weights_layer_2.at(i, j).unwrap();
                *self.weights_layer_2.at_mut(i, j).unwrap() = actual + epsilon;
                *gradient.weights_layer_2.at_mut(i, j).unwrap() =
                    (self.loss(training_input, training_output) - initial_loss) / epsilon;
                *self.weights_layer_2.at_mut(i, j).unwrap() = actual;
            }
        }

        for i in 0..self.bias_layer_2.rows {
            for j in 0..self.bias_layer_2.cols {
                let actual = *self.bias_layer_2.at(i, j).unwrap();
                *self.bias_layer_2.at_mut(i, j).unwrap() = actual + epsilon;
                *gradient.bias_layer_2.at_mut(i, j).unwrap() =
                    (self.loss(training_input, training_output) - initial_loss) / epsilon;
                *self.bias_layer_2.at_mut(i, j).unwrap() = actual;
            }
        }

        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, XorModel};

    #[test]
    fn xor() {
        let mut xor_model = XorModel::new();
        let training_input =
            Matrix::from_literal(vec![0f32, 0f32, 0f32, 1f32, 1f32, 0f32, 1f32, 1f32], 4, 2);
        let training_output = Matrix::from_literal(vec![0f32, 1f32, 1f32, 0f32], 4, 1);
        xor_model.train(&training_input, &training_output);
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
