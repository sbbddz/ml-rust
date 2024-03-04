use crate::matrix::Matrix;

pub trait Model {
    fn new() -> Self;
    fn forward(&self) -> Matrix;
    fn set_input_matrix(&mut self, input_matrix: Matrix);
    fn loss(&mut self, training_input: &Matrix, training_output: &Matrix) -> f32;
    fn train(&mut self, training_input: &Matrix, training_output: &Matrix) -> ();
    fn apply_diff(&mut self, diff: &Self, learning_rate: f32) -> ();
    fn finite_diff(
        &mut self,
        training_input: &Matrix,
        training_output: &Matrix,
        epsilon: f32,
    ) -> Self;
}

#[derive(Debug)]
struct BaseModel<'a> {
    input: Matrix,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    architecture: &'a [usize],
}

impl<'a> BaseModel<'a> {
    fn new(architecture: &'a [usize]) -> Self {
        let mut iter = architecture.iter();
        let mut input_val = iter.next().unwrap();
        let input = Matrix::new_rand(1, *input_val);
        let mut weights = vec![];
        let mut bias = vec![];

        for n in iter {
            weights.push(Matrix::new_rand(*input_val, *n));
            bias.push(Matrix::new_rand(1, *n));
            input_val = n;
        }

        BaseModel {
            input,
            weights,
            bias,
            architecture,
        }
    }

    fn forward(&self) -> Matrix {
        let mut weights_iter = self.weights.iter();
        let mut bias_iter = self.bias.iter();
        let first_weights = weights_iter.next().unwrap();
        let first_bias = bias_iter.next().unwrap();

        let mut activation_matrix = self.input.dot(first_weights);
        activation_matrix.sum(first_bias);
        activation_matrix.sigmoid();

        for w in weights_iter {
            let b = bias_iter.next().unwrap();
            let mut am = activation_matrix.dot(w);
            am.sum(b);
            am.sigmoid();
            activation_matrix = am;
        }

        activation_matrix
    }

    fn set_input_matrix(&mut self, input_matrix: Matrix) {
        assert!(self.input.shape() == input_matrix.shape());
        self.input = input_matrix;
    }

    fn loss(&mut self, training_input: &Matrix, training_output: &Matrix) -> f32 {
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

    fn train(&mut self, training_input: &Matrix, training_output: &Matrix) -> () {
        let loss = self.loss(training_input, training_output);
        println!("loss={loss}");

        for _ in 0..100 * 1000 {
            let diff = self.finite_diff(training_input, training_output, 0.1);
            self.apply_diff(&diff, 0.1);
            let loss = self.loss(training_input, training_output);
            println!("loss={loss}");
        }
    }

    fn apply_diff(&mut self, diff: &Self, learning_rate: f32) -> () {
        for (idx, w) in &mut self.weights.iter_mut().enumerate() {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    let actual_diff = *diff.weights.get(idx).unwrap().at(i, j).unwrap();
                    *w.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
                }
            }
        }

        for (idx, b) in &mut self.bias.iter_mut().enumerate() {
            for i in 0..b.rows {
                for j in 0..b.cols {
                    let actual_diff = *diff.bias.get(idx).unwrap().at(i, j).unwrap();
                    *b.at_mut(i, j).unwrap() -= learning_rate * actual_diff;
                }
            }
        }
    }

    fn finite_diff(
        &mut self,
        training_input: &Matrix,
        training_output: &Matrix,
        epsilon: f32,
    ) -> Self {
        let mut gradient = BaseModel::new(self.architecture);
        let initial_loss = self.loss(training_input, training_output);

        for (idx, w) in gradient.weights.iter_mut().enumerate() {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    let actual = *self.weights.get_mut(idx).unwrap().at(i, j).unwrap();
                    *self.weights.get_mut(idx).unwrap().at_mut(i, j).unwrap() = actual + epsilon;
                    *w.at_mut(i, j).unwrap() =
                        (self.loss(training_input, training_output) - initial_loss) / epsilon;
                    *self.weights.get_mut(idx).unwrap().at_mut(i, j).unwrap() = actual;
                }
            }
        }

        for (idx, b) in gradient.bias.iter_mut().enumerate() {
            for i in 0..b.rows {
                for j in 0..b.cols {
                    let actual = *self.bias.get_mut(idx).unwrap().at(i, j).unwrap();
                    *self.bias.get_mut(idx).unwrap().at_mut(i, j).unwrap() = actual + epsilon;
                    *b.at_mut(i, j).unwrap() =
                        (self.loss(training_input, training_output) - initial_loss) / epsilon;
                    *self.bias.get_mut(idx).unwrap().at_mut(i, j).unwrap() = actual;
                }
            }
        }

        gradient
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{BaseModel, Model};

    use super::Matrix;

    #[test]
    fn xor() {
        let mut xor_model = BaseModel::new(&[2, 3, 2, 1]);
        let training_input =
            Matrix::from_literal(vec![0f32, 0f32, 0f32, 1f32, 1f32, 0f32, 1f32, 1f32], 4, 2);
        let training_output = Matrix::from_literal(vec![0f32, 1f32, 1f32, 0f32], 4, 1);
        xor_model.train(&training_input, &training_output);
        println!();
        for i in 0..2 {
            for j in 0..2 {
                xor_model.set_input_matrix(Matrix::from_literal(vec![i as f32, j as f32], 1, 2));
                println!("{} | {} = {}", i, j, xor_model.forward())
            }
        }
    }
}
