// test unrolling and the helpfulness of it
use std::time::Instant;

use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx
};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // You can load a function from a pre-compiled PTX like so:
    let ptx = compile_ptx("
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = inp[i] * 3.0 + 2.0;
    }
}
    ").expect("Couldn't compile PTX");

    let module = ctx.load_module(ptx).expect("Can't load module from PTX");

    // and then load a function from it:
    let f = module.load_function("sin_kernel").unwrap();
    
    let mut a_host = vec![];
    for i in 0..1000 {
        a_host.push((i as f32)/100.0);
    }

    // we use a buidler pattern to launch kernels.
    let n = a_host.len() as i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&f);

    // timeit this section -- GPU
    let start_cuda = Instant::now(); 
    
    let a_dev = stream.memcpy_stod(&a_host)?;
    let mut b_dev = a_dev.clone(); // launch result kernel
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;
    let b_host = stream.memcpy_dtov(&b_dev)?; // get output

    let elapsed_cuda = start_cuda.elapsed();

    // timeit this section -- CPU
    let start_cpu = Instant::now();
    a_host.iter().map(|&a| a * 3.0 + 2.0).collect::<Vec<f32>>(); // perform computation
    let elapsed_cpu = start_cpu.elapsed();

    println!("Elapsed CUDA: {} ms", elapsed_cuda.as_secs_f64()*1000.0);
    println!("Elapsed CPU: {} ms", elapsed_cpu.as_secs_f64()*1000.0);

    Ok(())
}