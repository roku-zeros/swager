from db.models import *

res = []
with db_session:
    for c in Category.select():
        res.append(c.name)

    for c in AtmCategory.select():
        res.append(c.name)

print(list(set(res)))