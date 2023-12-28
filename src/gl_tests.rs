#[cfg(test)]
use super::{
    matrix::*,
    functions::*
};

#[test]
fn vector_mul_identity_matrix() {
    let vector = Vector4::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix4::identity().unwrap();

    assert_eq!(dot(&identity, &vector).unwrap(), vector);
}