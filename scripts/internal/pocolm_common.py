#!/usr/bin/env python3

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import sys
import subprocess
import time
from subprocess import CalledProcessError

# If the encoding of the default sys.stdout is not utf-8,
# force it to be utf-8. See PR #95.
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != "utf-8":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())


def ExitProgram(message):
    print("{0}: {1}".format(os.path.basename(sys.argv[0]), message),
          file=sys.stderr)
    # we exit in this way in case we're inside a thread in a multi-threaded
    # program; we want the entire program to exit.
    os._exit(1)


def RunCommand(command, log_file, verbose=False):
    if verbose:
        print("{0}: running command '{1}', log in {2}".format(
            os.path.basename(sys.argv[0]), command, log_file),
              file=sys.stderr)
    try:
        f = open(log_file, 'w', encoding="utf-8")
    except:
        ExitProgram('error opening log file {0} for writing'.format(log_file))

    # print the command to the log file.
    print('# {0}'.format(command), file=f)
    print('# running at ' + time.ctime(), file=f)
    f.flush()
    start_time = time.time()
    ret = subprocess.call(command,
                          shell=True,
                          stderr=f,
                          universal_newlines=True,
                          executable='/bin/bash')
    end_time = time.time()
    print('# exited with return code {0} after {1} seconds'.format(
        ret, '%.1f' % (end_time - start_time)),
          file=f)
    f.close()
    if ret != 0:
        ExitProgram(
            'command {0} exited with status {1}, output is in {2}'.format(
                command, ret, log_file))


def GetCommandStdout(command, log_file, verbose=False):
    if verbose:
        print("{0}: running command '{1}', log in {2}".format(
            os.path.basename(sys.argv[0]), command, log_file),
              file=sys.stderr)

    try:
        f = open(log_file, 'w', encoding="utf-8")
    except:
        ExitProgram('error opening log file {0} for writing'.format(log_file))

    # print the command to the log file.
    print('# ' + command, file=f)
    print('# running at ' + time.ctime(), file=f)
    start_time = time.time()
    try:
        output = subprocess.check_output(command,
                                         shell=True,
                                         stderr=f,
                                         universal_newlines=True,
                                         executable='/bin/bash')
    except CalledProcessError as e:
        end_time = time.time()
        print(e.output, file=f)
        print('# exited with return code {0} after {1} seconds'.format(
            e.returncode, '%.1f' % (end_time - start_time)),
              file=f)
        f.close()
        ExitProgram(
            'command {0} exited with status {1}, stderr is in {2} (output is: {3})'
            .format(command, e.returncode, log_file, e.output))

    print(output, file=f)
    end_time = time.time()
    print('# exited with return code 0 after {0} seconds'.format(
        '%.1f' % (end_time - start_time)),
          file=f)
    f.close()
    return output


def TouchFile(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a', encoding="utf-8").close()


def LogMessage(message):
    print(os.path.basename(sys.argv[0]) + ": " + message, file=sys.stderr)
