import coremltools as ct
import os

print(f"coremltools version: {ct.__version__}")
print("Converting to CoreML...")

# Teacher — ONNX load directly
print("Converting Teacher...")
teacher_model = ct.convert(
    '/Users/tariqulislam/bangla-dialect-models/teacher_fp32.onnx',
    source='milinternal',
    convert_to='mlprogram',
    compute_precision=ct.precision.FLOAT16,
)
teacher_model.save('/Users/tariqulislam/bangla-dialect-models/teacher_coreml.mlpackage')
print("Teacher saved!")

# Student
print("Converting Student...")
student_model = ct.convert(
    '/Users/tariqulislam/bangla-dialect-models/student_fp32.onnx',
    source='milinternal',
    convert_to='mlprogram',
    compute_precision=ct.precision.FLOAT16,
)
student_model.save('/Users/tariqulislam/bangla-dialect-models/student_coreml.mlpackage')
print("Student saved!")