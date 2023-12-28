use std::ops;

// Vector2

#[derive(Debug, PartialEq)]
#[repr(packed)]
pub struct Vector2(f32, f32);

impl ops::Add<Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, rhs: Vector2) -> Self::Output {
        Vector2::new((
            self.0 + rhs.0,
            self.1 + rhs.1
        ))
    }
}
impl ops::Sub<Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, rhs: Vector2) -> Self::Output {
        Vector2::new((
            self.0 - rhs.0,
            self.1 - rhs.1
        ))
    }
}
impl ops::Mul<f32> for Vector2 {
    type Output = Vector2;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector2::new((
            self.0 * rhs,
            self.1 * rhs
        ))
    }
}

impl Vector2 {
    pub fn new(dims: (f32, f32)) -> Self {
        Self (dims.0, dims.1)
    }

    pub fn length(&self) -> f32 {
        (
            self.0.powf(2.0) +
            self.1.powf(2.0)
        ).sqrt()
    }

    pub fn unit(&self) -> Vector2 {
        Vector2(
            self.0 / self.length(),
            self.1 / self.length()
        )
    }

    pub fn dot(&self, other: Vector2) -> f32 {
        self.0 * other.0 +
        self.1 * other.1
    }
}

// Vector3

#[derive(Debug, PartialEq)]
#[repr(packed)]
pub struct Vector3(f32, f32, f32);

impl ops::Add<Vector3> for Vector3 {
    type Output = Vector3;

    fn add(self, rhs: Vector3) -> Self::Output {
        Vector3::new((
            self.0 + rhs.0,
            self.1 + rhs.1,
            self.2 + rhs.2
        ))
    }
}
impl ops::Sub<Vector3> for Vector3 {
    type Output = Vector3;

    fn sub(self, rhs: Vector3) -> Self::Output {
        Vector3::new((
            self.0 - rhs.0,
            self.1 - rhs.1,
            self.2 - rhs.2
        ))
    }
}
impl ops::Mul<f32> for Vector3 {
    type Output = Vector3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector3::new((
            self.0 * rhs,
            self.1 * rhs,
            self.2 * rhs
        ))
    }
}

impl Vector3 {
    pub fn new(dims: (f32, f32, f32)) -> Self {
        Self (dims.0, dims.1, dims.2)
    }

    pub fn length(&self) -> f32 {
        (
            self.0.powf(2.0) +
            self.1.powf(2.0) +
            self.2.powf(2.0)
        ).sqrt()
    }

    pub fn unit(&self) -> Vector3 {
        Self(
            self.0 / self.length(),
            self.1 / self.length(),
            self.2 / self.length()
        )
    }

    pub fn dot(&self, other: Vector3) -> f32 {
        self.0 * other.0 +
        self.1 * other.1 +
        self.2 * other.2
    }

    // cross multiplication is only implemented for Vector3
    pub fn cross(&self, other: Vector3) -> Vector3 {
        Vector3(
            (self.1 * other.2) - (self.2 * other.1),
            (self.2 * other.0) - (self.0 * other.2),
            (self.0 * other.1) - (self.1 * other.0)
        )
    }
}

// Vector4

#[derive(Debug, PartialEq)]
#[repr(packed)]
pub struct Vector4(f32, f32, f32, f32);

impl ops::Add<Vector4> for Vector4 {
    type Output = Vector4;

    fn add(self, rhs: Vector4) -> Self::Output {
        Vector4::new((
            self.0 + rhs.0,
            self.1 + rhs.1,
            self.2 + rhs.2,
            self.3 + rhs.3
        ))
    }
}
impl ops::Sub<Vector4> for Vector4 {
    type Output = Vector4;

    fn sub(self, rhs: Vector4) -> Self::Output {
        Vector4::new((
            self.0 - rhs.0,
            self.1 - rhs.1,
            self.2 - rhs.2,
            self.3 - rhs.3
        ))
    }
}
impl ops::Mul<f32> for Vector4 {
    type Output = Vector4;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector4::new((
            self.0 * rhs,
            self.1 * rhs,
            self.2 * rhs,
            self.3 * rhs
        ))
    }
}

impl Vector4 {
    pub fn new(dims: (f32, f32, f32, f32)) -> Self {
        Self (dims.0, dims.1, dims.2, dims.3)
    }

    pub fn length(&self) -> f32 {
        (
            self.0.powf(2.0) +
            self.1.powf(2.0) +
            self.2.powf(2.0) +
            self.3.powf(2.0)
        ).sqrt()
    }

    pub fn unit(&self) -> Vector4 {
        Self(
            self.0 / self.length(),
            self.1 / self.length(),
            self.2 / self.length(),
            self.3 / self.length()
        )
    }

    pub fn dot(&self, other: Vector4) -> f32 {
        self.0 * other.0 +
        self.1 * other.1 +
        self.2 * other.2 +
        self.3 * other.3
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_2_operators() {
        let v1 = || Vector2::new((2.0, 3.0));
        let v2 = || Vector2::new((4.0, 2.0));

        assert_eq!(v1() + v2(), Vector2::new((6.0, 5.0)));
        assert_eq!(v1() - v2(), Vector2::new((-2.0, 1.0)));
        assert_eq!(v1() * 2.0, Vector2::new((4.0, 6.0)));

        assert_eq!(v1().dot(v2()), 14.0);
    }

    #[test]
    fn vector_2_length() {
        let vector = Vector2::new((3.0, 4.0));

        assert_eq!(vector.length(), 5.0);
    }

    #[test]
    fn vector_2_unit() {
        let vector = Vector2::new((3.0, 4.0));

        assert_eq!(vector.unit(), Vector2::new((0.6, 0.8)));
    }

    #[test]
    fn vector_3_operators() {
        let v1 = || Vector3::new((2.0, 3.0, 5.0));
        let v2 = || Vector3::new((4.0, 2.0, 2.0));

        assert_eq!(v1() + v2(), Vector3::new((6.0, 5.0, 7.0)));
        assert_eq!(v1() - v2(), Vector3::new((-2.0, 1.0, 3.0)));
        assert_eq!(v1() * 3.0, Vector3::new((6.0, 9.0, 15.0)));
        
        assert_eq!(v1().dot(v2()), 24.0);
        assert_eq!(v1().cross(v2()), Vector3::new((-4.0, 16.0, -8.0)));
    }

    #[test]
    fn vector_3_length() {
        let vector = Vector3::new((3.0, 4.0, 5.0));

        assert_eq!(vector.length(), 50.0_f32.sqrt());
    }

    #[test]
    fn vector_3_unit() {
        let vector = Vector3::new((2.0, 3.0, 4.0));

        assert_eq!(vector.unit(), Vector3::new(
            (
                2.0 / 29.0_f32.sqrt(),
                3.0 / 29.0_f32.sqrt(),
                4.0 / 29.0_f32.sqrt()
            )
        ));
    }

    #[test]
    fn vector_4_operators() {
        let v1 = || Vector4::new((2.0, 3.0, 5.0, 9.0));
        let v2 = || Vector4::new((4.0, 2.0, 2.0, 3.0));

        assert_eq!(v1() + v2(), Vector4::new((6.0, 5.0, 7.0, 12.0)));
        assert_eq!(v1() - v2(), Vector4::new((-2.0, 1.0, 3.0, 6.0)));
        assert_eq!(v1() * 4.0, Vector4::new((8.0, 12.0, 20.0, 36.0)));
        
        assert_eq!(v1().dot(v2()), 51.0);
    }

    #[test]
    fn vector_4_length() {
        let vector = Vector4::new((3.0, 4.0, 5.0, 1.0));

        assert_eq!(vector.length(), 51.0_f32.sqrt());
    }

    #[test]
    fn vector_4_unit() {
        let vector = Vector4::new((2.0, 3.0, 4.0, 5.0));

        assert_eq!(vector.unit(), Vector4::new(
            (
                2.0_f32.sqrt() / 27.0_f32.sqrt(),
                1.0 / 6.0_f32.sqrt(),
                8.0_f32.sqrt() / 27.0_f32.sqrt(),
                5.0 / 54.0_f32.sqrt()
            )
        ));
    }
}
