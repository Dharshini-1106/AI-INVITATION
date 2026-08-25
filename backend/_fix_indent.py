import io

p = 'app/core/understanding/parser.py'
s = open(p, encoding='utf-8').read()
lines = s.split('\n')
out = []
i = 0
n = len(lines)
while i < n:
    ls = lines[i]
    # Find the broken merge loop: the 'for k in (...' line that is NOT indented
    # (it follows the "# Merge matched" comment). We already have it indented
    # correctly in the file; just ensure the body lines are indented.
    if ls.strip().startswith('for k in ("event_name"'):
        out.append('    for k in ("event_name", "event_type", "bride_name", "groom_name", "date",')
        i += 1
        # consume the continuation line if present
        if i < n and 'time", "venue", "address", "contact_number"' in lines[i]:
            out.append('              "time", "venue", "address", "contact_number"):')
            i += 1
        # skip blanks
        while i < n and lines[i].strip() == '':
            out.append(lines[i]); i += 1
        if i < n and lines[i].strip().startswith('mv = matched_fields'):
            out.append('        mv = matched_fields.get(k, "")')
            i += 1
            while i < n and lines[i].strip() == '':
                out.append(lines[i]); i += 1
            if i < n and lines[i].strip().startswith('if mv'):
                out.append('        if mv:')
                i += 1
                while i < n and lines[i].strip() == '':
                    out.append(lines[i]); i += 1
                if i < n and lines[i].strip().startswith('primary['):
                    out.append('            primary[k] = mv')
                    i += 1
                continue
        continue
    out.append(ls)
    i += 1

open(p, 'w', encoding='utf-8').write('\n'.join(out))
print('done')
