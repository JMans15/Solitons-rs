#![allow(dead_code, unused_imports, non_snake_case)]

use std::{f64::consts::E, f64::consts::PI, path::Path, process::Command, process::Stdio};

use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy};
use plotters::{prelude::*, style::full_palette::GREEN_600};
use rayon::prelude::*;
use std::io::{stderr, Write};

struct Problem {
    dx: f64,
    dt: f64,
    delta: f64,
    u0: Array1<f64>,
    nx: usize,
    nt: usize,
    xl: f64,
    xr: f64,
    duration: f64,
    render_every: usize,
}

impl Problem {
    pub fn new(
        dx: f64,
        dt: f64,
        delta: f64,
        f0: impl Fn(&f64) -> f64,
        xl: f64,
        xr: f64,
        duration: f64,
        render_every: usize,
    ) -> Problem {
        let nx = ((xr - xl) / dx).floor() as usize;
        let nt = (duration / dt).floor() as usize;

        let x = Array::linspace(xl, xr, nx);

        Problem {
            dx,
            dt,
            delta,
            u0: x.map(f0),
            nx,
            nt,
            xl,
            xr,
            duration,
            render_every,
        }
    }
}

fn upwind_step(dx: f64, dt: f64, delta: f64, u: f64, ll: f64, l: f64, r: f64, rr: f64) -> f64 {
    let third = (delta / dx).powf(2 as f64) / dx / 2.0 * dt;
    let second = match u > 0.0 {
        true => u - l,
        false => r - u,
    };
    u - dt / dx * u * second - third * (-ll + 2.0 * l - 2.0 * r + rr)
}

fn step(
    dx: f64,
    dt: f64,
    delta: f64,
    u: f64,
    prev_u: f64,
    ll: f64,
    l: f64,
    r: f64,
    rr: f64,
) -> f64 {
    prev_u
        - (dt / dx) * (1. / 3.) * (r + u + l) * (r - l)
        - (delta / dx).powf(2.0) / dx * dt * (rr - 2. * r + 2. * l - ll)
}

fn precompute(problem: &Problem) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let Problem {
        dx,
        dt,
        delta,
        nx,
        nt,
        render_every: save_every,
        ..
    } = *problem;
    let u0 = problem.u0.clone();

    let num_saved = (nt as f32 / save_every as f32).floor() as usize;

    let mut grid = Array2::<f64>::zeros((num_saved, nx));

    let mut prevprev = Array1::<f64>::zeros(nx);
    let mut prev = Array1::<f64>::zeros(nx);
    let mut curr = Array1::<f64>::zeros(nx);

    prevprev.assign(&u0);
    grid.slice_mut(s![0, ..]).assign(&u0);

    prev.iter_mut().enumerate().for_each(|(k, elem)| {
        *elem = upwind_step(
            dx,
            dt,
            delta,
            prevprev[k],
            prevprev[(k as i32 - 2).rem_euclid(nx as i32) as usize],
            prevprev[(k as i32 - 1).rem_euclid(nx as i32) as usize],
            prevprev[(k + 1).rem_euclid(nx)],
            prevprev[(k + 2).rem_euclid(nx)],
        );
    });

    let mut stderr = stderr();
    let mut nsaved: usize = 1;
    for s in 1..nt - 1 {
        curr.iter_mut().enumerate().for_each(|(k, elem)| {
            *elem = step(
                dx,
                dt,
                delta,
                prev[k],
                prevprev[k],
                prev[(k as i32 - 2).rem_euclid(nx as i32) as usize],
                prev[(k as i32 - 1).rem_euclid(nx as i32) as usize],
                prev[(k + 1).rem_euclid(nx)],
                prev[(k + 2).rem_euclid(nx)],
            );
        });
        if s % save_every == 0 && nsaved < num_saved {
            grid.slice_mut(s![nsaved, ..]).assign(&curr);
            nsaved += 1;
            eprint!("\rComputed until t = {:.2}", (s + 2) as f64 * dt);
            stderr.flush().unwrap();
        }
        prevprev.assign(&prev);
        prev.assign(&curr);
    }
    eprintln!("\r== Computation finished ==");
    stderr.flush().unwrap();
    Ok(grid)
}

fn sech(x: f64) -> f64 {
    2.0f64 / (E.powf(x) + E.powf(-x))
}

fn initial_soliton(u0: f64, x0: f64, uinf: f64, delta: f64, x: &f64) -> f64 {
    #[allow(non_snake_case)]
    let Delta = delta * ((u0 - uinf) / 12.).powf(-0.5);

    uinf + (u0 - uinf) * sech((x - x0) / Delta).powf(2.0)
}

fn render_images(
    grid: Array2<f64>,
    problem: &Problem,
    yu: f32,
    yd: f32,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = Command::new("rm").args(["-rf", "./.results"]).status()?;
    assert!(output.success());
    output = Command::new("mkdir").args([".results"]).status()?;
    assert!(output.success());

    let render_vectors = grid
        .into_raw_vec()
        .chunks(problem.nx as usize)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    render_vectors
        .into_par_iter()
        .enumerate()
        .for_each(|(j, elem)| {
            let filename = format!(".results/img_{:05}.png", j + 1);

            let root = BitMapBackend::new(&filename, (width, height)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let (upper, lower) = root.split_vertically(height - 25);

            let mut chart = ChartBuilder::on(&upper)
                .caption(
                    "Three catching solitons, Zabusky and Kruskal scheme",
                    ("sans-serif", 20),
                )
                .margin_left(10)
                .margin_right(10)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(problem.xl as f32..problem.xr as f32, yd..yu)
                .unwrap();

            chart
                .configure_mesh()
                .y_desc("u(x)")
                .x_desc("x")
                .bold_line_style(&BLACK.mix(0.2))
                .light_line_style(&WHITE)
                .draw()
                .unwrap();

            chart
                .draw_series(
                    AreaSeries::new(
                        (0..problem.nx.clone()).map(|i| {
                            (
                                (i as f32 * problem.dx as f32) + problem.xl as f32,
                                elem[i as usize] as f32,
                            )
                        }),
                        -2.0,
                        &BLUE.mix(0.2),
                    )
                    .border_style(Into::<ShapeStyle>::into(&BLUE).stroke_width(2)),
                )
                .unwrap()
                .label(format!(
                    "u at time {:.2}",
                    j as f64 * problem.render_every as f64 * problem.dt
                ))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

            chart
                .draw_series(LineSeries::new(
                    (0..problem.nx).map(|i| {
                        (
                            (i as f32 * problem.dx as f32) + problem.xl as f32,
                            problem.u0[i as usize] as f32,
                        )
                    }),
                    &GREEN_600,
                ))
                .unwrap()
                .label("Initial function")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN_600));

            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()
                .unwrap();

            lower
                .titled(
                    format!(
                        "dt = {:.3e}, dx = {:.3}, delta = {:.3}, t = {:.3}",
                        problem.dt,
                        problem.dx,
                        problem.delta,
                        j as f64 * problem.render_every as f64 * problem.dt as f64
                    )
                    .as_str(),
                    ("sans-serif", 20, &BLACK.mix(0.5)).into_text_style(&root),
                )
                .unwrap();

            root.present().unwrap();
            //println!("Rendered frame {}", j);
        });
    Ok(())
}

fn create_gif() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = Command::new("rm").arg("-f").arg("./result.gif").status()?;
    assert!(output.success());
    eprintln!("Generating gif from images");
    output = Command::new("ffmpeg")
        .arg("-i")
        .arg(".results/img_%05d.png")
        .arg("-vf")
        .arg("fps=30,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
        .arg("result.gif")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("Couldn't create gif");
    assert!(output.success());
    eprintln!("Removing intermediate images");
    output = Command::new("rm").args(["-rf", "./.results"]).status()?;
    assert!(output.success());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Condition initiale
    // Alternative: |x| f64::cos(PI*x)
    let f = |x: &f64| -> f64 {
        initial_soliton(1., -0.5, 0., 0.022, x)
            + initial_soliton(0.5, 0., 0., 0.022, x)
            + initial_soliton(0.25, 0.5, 0., 0.022, x)
    };

    // let f = |x: &f64| -> f64 { f64::cos(PI * x) };
    let problem = Problem::new(
        0.005,
        1e-5 * 1. / 30.,
        0.022,
        &f,
        -1.0f64,
        1.0f64,
        10.,
        1e5 as usize,
    );
    let grid = precompute(&problem)?;

    // Crée une image par frame dans le dossier .results
    eprintln!("Rendering images");
    render_images(grid, &problem, 1.2f32, -0.2f32, 1200, 600)?;

    // Lis le dossier .results, créée un gif avec les image set supprime le dossier
    create_gif()?;

    Ok(())
}
