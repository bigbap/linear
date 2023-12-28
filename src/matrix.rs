use std::{ops, usize};

use super::{
    errors::MatrixError,
    functions::*
};

pub trait TransformationMatrices<const R: usize, const C: usize> {
    fn identity() -> Result<Matrix<R, C>, MatrixError> {
        let vector = Vector::<R>::new(vec![1.0; R])?;

        Self::scaling_matrix(vector)
    }
    fn scaling_matrix(vector: Vector<R>) -> Result<Matrix<R, C>, MatrixError> {
        let mut data = Vec::<f32>::new();

        for row in 0..R {
            for col in 0..C {
                data.push(match row == col {
                    true => vector.get(row, 0)?,
                    _ => 0.0
                });
            }
        }

        Matrix::<R, C>::new(data)
    }
    fn translation_matrix(vector: Vector<R>) -> Result<Matrix<R, C>, MatrixError> {
        let mut matrix = Self::identity()?;
        
        for row in 0..R {
            let index = (row * C) + C - 1;

            matrix.data[index] = vector.get(row, 0)?;
        }

        Ok(matrix)
    }
}

#[derive(Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize> {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

pub type Matrix2 = Matrix::<2, 2>;
pub type Matrix3 = Matrix::<3, 3>;
pub type Matrix4 = Matrix::<4, 4>;
impl TransformationMatrices<2, 2> for Matrix2 {}
impl TransformationMatrices<3, 3> for Matrix3 {}
impl TransformationMatrices<4, 4> for Matrix4 {}

pub type Vector<const R: usize> = Matrix::<R, 1>;
pub type Vector2 = Matrix::<2, 1>;
pub type Vector3 = Matrix::<3, 1>;
pub type Vector4 = Matrix::<4, 1>;

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub fn new(data: Vec<f32>) -> Result<Matrix<R, C>, MatrixError> {
        if data.is_empty() {
            return Err(MatrixError::EmptyData);
        }

        if data.len() != R * C {
            return Err(MatrixError::TheDataDoesNotMatchTheDimensions);
        }

        Ok(Matrix {
            data,
            rows: R,
            cols: C
        })
    }

    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f32, MatrixError> {
        let index = (row * self.cols) + col;
    
        if index >= self.data.len() {
            return Err(MatrixError::OutOfBounds);
        }
    
        Ok(self.data[index])
    }
}

impl<const R: usize, const C: usize> ops::Add<Matrix<R, C>> for Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn add(self, rhs: Matrix<R, C>) -> Self::Output {
        if self.data.len() != rhs.data.len()  {
            panic!("lhs and rhs matrices must have the same dimensions");
        }
    
        let mut result: Vec<f32> = Vec::new();
        for (i, _) in self.data.iter().enumerate() {
            result.push(self.data[i] + rhs.data[i]);
        }

        Matrix::<R, C>::new(result).unwrap()
    }
}

impl<const R: usize, const C: usize> ops::Sub<Matrix<R, C>> for Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn sub(self, rhs: Matrix<R, C>) -> Self::Output {
        if self.data.len() != rhs.data.len()  {
            panic!("lhs and rhs matrices must have the same dimensions");
        }
    
        let mut result: Vec<f32> = Vec::new();
        for (i, _) in self.data.iter().enumerate() {
            result.push(self.data[i] - rhs.data[i]);
        }

        Matrix::<R, C>::new(result).unwrap()
    }
}

// dot product
impl<const A: usize, const B: usize, const C: usize> ops::Mul<&Matrix<B, C>> for &Matrix<A, B> {
    type Output = Matrix<A, C>;
    
    fn mul(self, rhs: &Matrix<B, C>) -> Self::Output {
        dot(self, rhs).unwrap()
    }
}

impl<const R: usize, const C: usize> ops::Mul<f32> for &Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut result: Vec<f32> = Vec::new();
        for (i, _) in self.data.iter().enumerate() {
            result.push(self.data[i] * rhs);
        }

        Matrix::<R, C>::new(result).unwrap()
    }
}
