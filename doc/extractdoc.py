from __future__ import print_function
import os
import sys

if len(sys.argv) < 4:
    sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]
with open(infile, 'r') as f:
    instring = f.read()

macros = []
for ii, line in enumerate(instring.split('\n')):
    if line[:7] == '#define':
        if len(line.split('#define ')[1].split()) < 2:
            continue
        macros.append((line.split('#define ')[1].split()[0], line.split('#define ')[1].split('(')[1].split(')')[0]))

functions = []

while '/**' in instring:
    functionanddoc = instring.split('\n/**')[1].split(');\n')[0]
    instring = instring.split(functionanddoc)[1]

    functionanddoc = '/* ' + functionanddoc
    
    function = " ".join(functionanddoc.split('*/\n')[1].strip().split()) + ')'

    doc = '\n'.join(['   '+l[2:] for l in functionanddoc.split('*/\n')[0].strip().split('\n')])

    doc = doc.replace('trlib_int_t', ':c:type:`trlib_int_t`')
    doc = doc.replace('trlib_flt_t', ':c:type:`trlib_flt_t`')

    functions.append((function, doc))

outlines = [sys.argv[3], '='.join(['' for ii in range(1+len(sys.argv[3]))]), '', '', 'Functions', '------------', '', '']

for fun in functions:
    outlines.append('.. c:function:: {:s}\n\n{:s}\n\n'.format(fun[0], fun[1]))

outlines = outlines + ['', '', 'Definitions', '------------', '', '']

for macro in macros:
    outlines.append('.. c:macro:: {:s}\n\n  {:s}\n\n'.format(macro[0], macro[1]))

with open(outfile, 'w') as f:
    f.write('\n'.join(outlines))
