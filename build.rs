//! Compile qhull (extern/qhull, the version matplotlib 3.11 builds) for the
//! Delaunay triangulations of `matplotlib_rs._qhull`.

fn main() {
    let files = [
        "geom2_r.c", "geom_r.c", "global_r.c", "io_r.c", "libqhull_r.c", "mem_r.c",
        "merge_r.c", "poly2_r.c", "poly_r.c", "qset_r.c", "random_r.c", "rboxlib_r.c",
        "stat_r.c", "usermem_r.c", "userprintf_rbox_r.c", "userprintf_r.c", "user_r.c",
    ];
    let mut build = cc::Build::new();
    build.include("extern/qhull/src").warnings(false);
    for f in files {
        build.file(format!("extern/qhull/src/libqhull_r/{f}"));
    }
    build.file("extern/qhull/mplrs_delaunay.c");
    build.compile("mplrs_qhull");
    println!("cargo:rerun-if-changed=extern/qhull");
}
