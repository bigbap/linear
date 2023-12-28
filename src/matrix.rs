use std::{ops, usize};
use super::errors::MatrixError;

pub trait Identity<T> {
    fn identity() -> Result<T, MatrixError>;
}

#[derive(Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize> {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

pub type Matrix2 = Matrix::<2, 2>;
impl Identity<Matrix2> for Matrix2 {
    fn identity() -> Result<Matrix2, MatrixError> {
        Matrix2::new(vec![
            1.0, 0.0,
            0.0, 1.0
        ])
    }
}
pub type Matrix3 = Matrix::<3, 3>;
impl Identity<Matrix3> for Matrix3 {
    fn identity() -> Result<Matrix3, MatrixError> {
        Matrix3::new(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ])
    }
}
pub type Matrix4 = Matrix::<4, 4>;
impl Identity<Matrix4> for Matrix4 {
    fn identity() -> Result<Matrix4, MatrixError> {
        Matrix4::new(vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    }
}

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

impl<const R: usize, const C: usize> ops::Mul<f32> for Matrix<R, C> {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut result: Vec<f32> = Vec::new();
        for (i, _) in self.data.iter().enumerate() {
            result.push(self.data[i] * rhs);
        }

        Matrix::<R, C>::new(result).unwrap()
    }
}
