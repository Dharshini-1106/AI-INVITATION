import io
p = 'app/core/understanding/parser.py'
s = open(p, encoding='utf-8').read()
# Fix indentation of the housewarming line in EVENT_TYPE_KEYWORDS
s = s.replace('"namingceremony": ("Naming Ceremony", 0.8),\n"housewarming":',
              '"namingceremony": ("Naming Ceremony", 0.8),\n    "housewarming":')
open(p, 'w', encoding='utf-8').write(s)
import ast
ast.parse(s)
print('indent fixed + syntax OK')
