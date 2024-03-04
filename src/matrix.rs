use core::fmt;

use crate::get_random_float;

#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
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
