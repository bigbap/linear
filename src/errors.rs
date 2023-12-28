#[derive(Debug, PartialEq, thiserror::Error)]
pub enum MatrixError {
    #[error("cannot create a matrix with empty data")]
    EmptyData,

    #[error("the matrix is not a vector")]
    NotAVector,

    #[error("location is out of bounds")]
    OutOfBounds,

    #[error("the provided data does not match the matrix dimensions")]
    TheDataDoesNotMatchTheDimensions,

    #[error("lhs and rhs matrices must have the same dimensions")]
    MatricesMustHaveSameDimensions,

    #[error("lhs and rhs matrices do not have matching dimensions for dot product")]
    MatrixDimensionsDoNotMatchForDotProduct,
    
    #[error("lhs and rhs must be vector_3 for cross product")]
    VectorMustBe3DForCrossProduct
}