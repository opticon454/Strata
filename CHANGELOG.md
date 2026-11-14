# Changelog

## 20261110-1543

- Used the CUDA half2 raw representation for bit-preserving lane conversion on CUDA 12 compilers.

## 20261110-1535

- Added exact FP32 handling for the Q2 weight-stationary integer half-lanes on pre-sm_53 CUDA targets, retaining newer devices' PTX.
- Checked those production helpers in the bounded Maxwell arithmetic test.

## 20261110-1528

- Aligned the experimental runtime capability floor with its existing sm_52 build support while retaining the normal release floor.
- Included the real device memory self-check in the Maxwell qualification preset.

## 20261110-1519

- Added an opt-in bounded GPU arithmetic parity check for experimental Maxwell builds without model loads or timing loops.

- Added an explicit CUDA 12 sm_52 build preset while preserving the released build defaults.
