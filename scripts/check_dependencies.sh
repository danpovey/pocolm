#!/bin/bash

# Apache 2.0.  Copyright 2011-2016 Johns Hopkins University (author: Daniel
# Povey)

# this script is based on the check_dependencies.sh script in Kaldi.

# at some point we could try to add packages for Cywgin or macports(?) to this
# script.
redhat_packages=
debian_packages=
opensuse_packages=

function add_packages {
  redhat_packages="$redhat_packages $1";
  debian_packages="$debian_packages $2";
  opensuse_packages="$opensuse_packages $3";
}

status=0

if ! which g++ >&/dev/null; then
  echo "$0: g++ is not installed."
  add_packages gcc-c++ g++ gcc-c++
fi

for f in make gcc grep gzip git bash; do
  if ! which $f >&/dev/null; then
    echo "$0: $f is not installed."
    add_packages $f $f $f
  fi
done

if ! which svn >&/dev/null; then
  echo "$0: subversion is not installed"
  add_packages subversion subversion subversion
fi

if ! which awk >&/dev/null; then
  echo "$0: awk is not installed"
  add_packages gawk gawk gawk
fi

if which python >&/dev/null ; then
  version=`/usr/bin/env python 2>&1 --version | awk '{print $2}' `
  if [[ $version != "2.7"* && $version != "3."* ]] ; then
    status=1
    if which python2.7 >&/dev/null ; then
      echo "$0: python 2.7 is not the default python (lower version python does not "
      echo "$0: have packages that are required by pocolm). You should make it default"
    else
      echo "$0: python 2.7 is not installed"
      add_packages python2.7 python2.7 python2.7
    fi
  fi

else
  echo "$0: python 2.7 is not installed"
  add_packages python2.7 python2.7 python2.7
fi

if ! python -c 'import numpy' >&/dev/null; then
  echo "$0: python-numpy is not installed"
  # I'm not sure if this package name is OK for all distributions, this is what
  # it seems to be called on Debian.  We'll have to investigate this.
  add_packages numpy python3-numpy python3-numpy
fi

printed=false

if which apt-get >&/dev/null && ! which zypper >/dev/null; then
  # if we're using apt-get [but we're not OpenSuse, which uses zypper as the
  # primary installer, but sometimes installs apt-get for some compatibility
  # reason without it really working]...
  if [ ! -z "$debian_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo apt-get install $debian_packages"
    printed=true
    status=1
  fi
  if ! dpkg -l | grep -E 'libatlas3gf|libatlas3-base' >/dev/null; then
    echo "You should probably do: "
    echo " sudo apt-get install libatlas3-base"
    printed=true
  fi
fi

redhat_pkg_mgr=
if which dnf >&/dev/null; then
  redhat_pkg_mgr=dnf
elif which yum >&/dev/null; then
  redhat_pkg_mgr=yum
fi
if [ -n "$redhat_pkg_mgr" ]; then
  if [ ! -z "$redhat_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo $redhat_pkg_mgr install $redhat_packages"
    printed=true
    status=1
  fi
  if ! rpm -qa|  grep atlas >/dev/null; then
    echo "You should probably do something like: "
    echo "sudo $redhat_pkg_mgr install atlas.x86_64"
    printed=true
  fi
fi

if which zypper >&/dev/null; then
  if [ ! -z "$opensuse_packages" ]; then
    echo "$0: we recommend that you run (our best guess):"
    echo " sudo zypper install $opensuse_packages"
    printed=true
    status=1
  fi
  if ! zypper search -i | grep -E 'libatlas3|libatlas3-devel' >/dev/null; then
    echo "You should probably do: "
    echo "sudo zypper install libatlas3-devel"
    printed=true
  fi
fi

if [ ! -z "$debian_packages" ]; then
  # If the list of packages to be installed is nonempty,
  # we'll exit with error status.  Check this outside of
  # hecking for yum or apt-get, as we want it to exit with
  # error even if we're not on Debian or red hat.
  status=1
fi


if [ $(pwd | wc -w) -gt 1 ]; then
  echo "*** $0: Warning: pocolm scripts may fail if the directory name contains a space."
  echo "***  (it's OK if you just want to compile a few tools -> disable this check)."
  status=1;
fi

if ! $printed && [ $status -eq 0 ]; then
  echo "$0: all OK."
fi


exit $status
