use std::{f64::consts::PI, f64::consts::E, process::Command, path::Path};

use plotters::{prelude::*, style::full_palette::GREEN_600};
use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy};
use rayon::prelude::*;

fn upwind_step(dx: f64, dt: f64, delta: f64, u: f64,
               ll: f64, l: f64, r: f64, rr: f64) -> f64
{
    let third = (delta/dx).powf(2 as f64)/dx/2.0*dt;
    let second = match u > 0.0 {
        true => u-l,
        false => r-u,
    };
    u - dt/dx * u * second - third * (-ll+2.0*l-2.0*r+rr)
}

fn step(dx: f64, dt: f64, delta: f64, u: f64, prev_u: f64,
               ll: f64, l: f64, r: f64, rr: f64) -> f64
{
    prev_u - dt/dx/3. * (r+u+l)*(r-l) - (delta/dx).powf(2.)/dx*dt * (rr-2.*r+2.*l-ll)
}

fn precompute(nt: usize, nx: usize, u0: Array1<f64>, dx: f64, dt: f64, delta: f64) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let mut grid = Array2::<f64>::zeros((nt,nx));
    grid.slice_mut(s![0,..]).assign(&u0);

    for k in 0..nx {
        grid[[1, k]] = upwind_step(
               dx, dt, delta,
               grid[[0, k]],
               grid[[0, (k as i32 - 2).rem_euclid(nx as i32) as usize]] ,
               grid[[0, (k as i32 -1).rem_euclid(nx as i32) as usize]],
               grid[[0, (k+1).rem_euclid(nx)]],
               grid[[0, (k+2).rem_euclid(nx)]]
            );
    }

    for s in 1..nt-1 {
        for k in 0..nx {
           grid[[s+1, k]] = step(
               dx, dt, delta,
               grid[[s, k]],
               grid[[s-1, k]],
               grid[[s, (k as i32 - 2).rem_euclid(nx as i32) as usize]] ,
               grid[[s, (k as i32 -1).rem_euclid(nx as i32) as usize]],
               grid[[s, (k+1).rem_euclid(nx)]],
               grid[[s, (k+2).rem_euclid(nx)]]
            );
        }
        match s%1000 {
            0 => eprintln!("Computed until t = {:.2}", (s+2) as f64*dt),
            _ => (),
        };
    }
    Ok(grid)
}

fn sech(x: f64) -> f64 {
    2 as f64 / (E.powf(x) + E.powf(-x))
}

fn initial_soliton(amplitude: f64, position: f64, x: &f64) -> f64 {
    let k: f64 = (amplitude/2.).sqrt();
    let nu = -position*k;

    amplitude*sech(k*x+nu).powf(2.)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulation parameters
    let dx = 0.01f64;
    let dt = 0.000005f64;
    let delta = 0.022f64;

    // View parameters
    let xl = 0.0f64;
    let xr = 2.0f64;
    let duration: f64 = 5.;
    let render_every: usize = 5000;

    // Deriving the number of steps
    let nx = ((xr-xl)/dx).ceil() as i32;
    let nt = (duration/dt).ceil() as i32;

    // Initial condition and domain
    let x = Array::linspace(xl, xr, nx as usize);
    let u0 = x.map(|x| f64::cos(PI*x));

    //let u0 = x.map(|x| initial_soliton(1., 0., x) + initial_soliton(4., -5., x));

    eprintln!("Saving the result in a npy file");
    let saved_path = Path::new("grid.npy");

    let grid: Array2<f64> = match saved_path.exists() {
        true => read_npy("grid.npy")?,
        false => precompute(nt as usize, nx as usize, u0, dx, dt, delta)?
    };

    if !saved_path.exists() {
        write_npy("grid.npy", &grid)?;
    }

    // Creating images in parallel
    let mut output = Command::new("rm").args(["-rf", "./.results"]).status()?;
    assert!(output.success());
    output = Command::new("mkdir").args([".results"]).status()?;
    assert!(output.success());

    let render_size = grid.slice(s![..;render_every,..]).nrows();
    let mut render_grid = Array2::<f64>::zeros((render_size, nx as usize));

    render_grid.assign(&grid.slice(s![..;render_every,..]));
    let render_vectors = render_grid.into_raw_vec().chunks(nx as usize).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

    render_vectors.into_par_iter().enumerate().for_each(|(j, elem)| {
        let filename = format!(".results/img_{:04}.png", j+1);

        let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption("Solitons, Zabusky and Kruskal scheme", ("sans-serif", 20))
            .build_cartesian_2d(xl as f32..xr as f32, -1.1f32..2.5f32).unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                (0..nx)
                .map(|i| ((i as f32 * dx as f32) + xl as f32, elem[i as usize] as f32)),
                &RED
            )).unwrap()
            .label(format!("u at time {:.2}", j as f64 * render_every as f64 * dt))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], &RED));

        chart
            .draw_series(LineSeries::new(
                (0..nx)
                .map(|i| ((i as f32 * dx as f32) + xl as f32, grid[[0, i as usize]] as f32)),
                &GREEN_600
            )).unwrap()
            .label("cos(pi*x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], &GREEN_600));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw().unwrap();

        root.present().unwrap();
        println!("Rendered frame {}", j);

    });

    // Combining the images into a gif
    output = Command::new("rm").arg("-f").arg("./result.gif").status()?;
    assert!(output.success());
    eprintln!("Generating gif from images");
    output = Command::new("ffmpeg")
        .arg("-i").arg(".results/img_%04d.png")
        .arg("-vf").arg("fps=30,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
        .arg("result.gif")
        .status()?;
    assert!(output.success());
    eprintln!("Removing intermediate images");
    output = Command::new("rm").args(["-rf", "./.results"]).status()?;
    assert!(output.success());

    Ok(())
}
