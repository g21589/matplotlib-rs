/*
 * Delaunay triangulation with qhull, as matplotlib's src/_qhull_wrapper.cpp
 * (delaunay_impl) computes it, so that matplotlib_rs's triangulations,
 * including the order of the triangles and of their vertices, are identical.
 * Called from src/python/tri.rs, which checks the arguments.
 */
#include "libqhull_r/qhull_ra.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define MPLRS_DEVNULL "NUL"
#else
#define MPLRS_DEVNULL "/dev/null"
#endif

const char *mplrs_qhull_version(void)
{
    return qh_version;
}

void mplrs_qhull_free(int *p)
{
    free(p);
}

/*
 * Triangulate the npoints points (x, y).  Returns qhull's exit code (0 on
 * success), or -1 if memory or the null device could not be allocated.  On
 * success *ntri is the number of triangles and *triangles, *neighbors are
 * arrays of 3 * *ntri ints, to be freed with mplrs_qhull_free.  *leaked is
 * set if qhull could not free all its memory.  qhull's error messages are
 * written to stderr unless hide_errors is set.
 */
int mplrs_qhull_delaunay(int npoints, const double *x, const double *y, int hide_errors,
                         int *ntri, int **triangles, int **neighbors, int *leaked)
{
    qhT qh_qh;
    qhT *qh = &qh_qh;
    facetT *facet;
    vertexT *vertex, **vertexp;
    facetT *neighbor, **neighborp;
    int i, n, max_facet_id, exitcode, curlong, totlong;
    double x_mean = 0.0, y_mean = 0.0;
    coordT *points;
    int *tri_indices = NULL, *tri = NULL, *nbr = NULL;
    FILE *error_file;

    QHULL_LIB_CHECK

    *ntri = 0;
    *triangles = *neighbors = NULL;
    *leaked = 0;
    points = (coordT *)malloc(sizeof(coordT) * 2 * (size_t)npoints);
    if (points == NULL)
        return -1;
    /* Points are centered on their mean, as by matplotlib. */
    for (i = 0; i < npoints; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= npoints;
    y_mean /= npoints;
    for (i = 0; i < npoints; ++i) {
        points[2*i] = x[i] - x_mean;
        points[2*i+1] = y[i] - y_mean;
    }

    error_file = hide_errors ? fopen(MPLRS_DEVNULL, "w") : stderr;
    if (error_file == NULL) {
        free(points);
        return -1;
    }

    qh_zero(qh, error_file);
    exitcode = qh_new_qhull(qh, 2, npoints, points, False,
                            (char *)"qhull d Qt Qbb Qc Qz", NULL, error_file);
    if (exitcode == qh_ERRnone) {
        /* Split facets so that they only have 3 points each. */
        qh_triangulate(qh);
        n = 0;
        FORALLfacets {
            if (!facet->upperdelaunay)
                ++n;
        }
        max_facet_id = qh->facet_id - 1;
        tri_indices = (int *)malloc(sizeof(int) * (size_t)(max_facet_id + 1));
        tri = (int *)malloc(sizeof(int) * 3 * (size_t)(n > 0 ? n : 1));
        nbr = (int *)malloc(sizeof(int) * 3 * (size_t)(n > 0 ? n : 1));
        if (tri_indices == NULL || tri == NULL || nbr == NULL) {
            free(tri);
            free(nbr);
            tri = nbr = NULL;
            exitcode = -1;
        }
        else {
            int *t = tri, *b = nbr;
            i = 0;
            FORALLfacets {
                if (!facet->upperdelaunay) {
                    int indices[3], *ip = indices;
                    tri_indices[facet->id] = i++;
                    FOREACHvertex_(facet->vertices) {
                        *ip++ = qh_pointid(qh, vertex->point);
                    }
                    *t++ = (facet->toporient ? indices[0] : indices[2]);
                    *t++ = indices[1];
                    *t++ = (facet->toporient ? indices[2] : indices[0]);
                }
                else {
                    tri_indices[facet->id] = -1;
                }
            }
            FORALLfacets {
                if (!facet->upperdelaunay) {
                    int indices[3], *ip = indices;
                    FOREACHneighbor_(facet) {
                        *ip++ = (neighbor->upperdelaunay ? -1 : tri_indices[neighbor->id]);
                    }
                    *b++ = (facet->toporient ? indices[2] : indices[0]);
                    *b++ = (facet->toporient ? indices[0] : indices[2]);
                    *b++ = indices[1];
                }
            }
            *ntri = n;
        }
        free(tri_indices);
    }

    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (curlong || totlong)
        *leaked = 1;
    if (error_file != stderr)
        fclose(error_file);
    free(points);
    if (exitcode == qh_ERRnone) {
        *triangles = tri;
        *neighbors = nbr;
    }
    return exitcode;
}
