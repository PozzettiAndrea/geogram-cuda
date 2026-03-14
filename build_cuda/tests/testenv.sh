#!/bin/sh
# Template to set environment variables for the
# non-regression testing framework.

export VORPALINE_BUILD_CONFIG="Release"
export VORPALINE_SOURCE_DIR="/home/shadeform/cudageom/geogram-cuda"
export VORPALINE_BUILD_DIR="/home/shadeform/cudageom/geogram-cuda/build_cuda"
export VORPALINE_BIN_DIR="/home/shadeform/cudageom/geogram-cuda/build_cuda/bin"
export VORPALINE_LIB_DIR="/home/shadeform/cudageom/geogram-cuda/build_cuda/lib"
export VORPATEST_ROOT_DIR="/home/shadeform/cudageom/geogram-cuda/tests"
export DATADIR="/home/shadeform/cudageom/geogram-cuda/tests/data"

args=
while [ -n "$1" ]; do
    case "$1" in
        --with-*=*)
            var=`echo "$1" | sed 's/--with-\([^=]*\)=\(.*\)$/VORPALINE_WITH_\U\1\E=\2/'`
            export "$var"
            shift
            ;;
        --with-*)
            var=`echo "$1" | sed 's/--with-\(.*\)$/VORPALINE_WITH_\U\1=1/'`
            export "$var"
            shift
            ;;
        *)
            args="$args $1"
            shift;
            ;;
    esac
done

