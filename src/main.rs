use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

const SRC: &str = "
#define N 1000
extern \"C\" __global__ void add_vectors(float *a, float *b, float *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}
";

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();
    let ptx = compile_ptx(SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "add_vectors", &["add_vectors"])?;
    let f = dev.get_func("add_vectors", "add_vectors").unwrap();
    println!("Loaded in {:?}", start.elapsed());

    let ah = [1.0f32, 10., 150.];
    let bh = [1.0f32, 10., -50.];
    let mut ch = [0.0f32; 3];
    let a_dev = dev.htod_sync_copy(&ah)?;
    let b_dev = dev.htod_sync_copy(&bh)?;
    let mut c_dev = dev.htod_sync_copy(&ch)?;

    println!("Copied in {:?}", start.elapsed());
    let cfg = LaunchConfig {
        block_dim: (3, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let _res = unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev)) };
    dev.dtoh_sync_copy_into(&c_dev, &mut ch)?;
    println!("Adding vectors {:?} and {:?}.", ah, bh);
    println!("Found {:?} in {:?}", ch, start.elapsed());
    Ok(())
}