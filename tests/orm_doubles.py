"""Dobles ligeros de mongoengine para probar los módulos que hablan con el ORM.

Los módulos del pipeline usan `ganabosques_orm` con el estilo habitual de
mongoengine: `Documento.objects(filtro).first()`, `.count()`, `.delete()`,
`Documento.objects.insert(lista)` y `documento.save()`. Estas clases replican
esa superficie sobre listas en memoria, de modo que las pruebas ejercitan la
lógica real de agregación y guardado sin necesidad de una base de datos.
"""

from typing import Any, Dict, List


class FakeQuerySet:
    """Resultado de una consulta: itera, cuenta, ordena y borra en memoria."""

    def __init__(self, items, store=None):
        self._items = list(items)
        self._store = store

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def first(self):
        return self._items[0] if self._items else None

    def count(self):
        return len(self._items)

    def only(self, *fields):
        return self

    def no_cache(self):
        return self

    def order_by(self, *fields):
        if not fields or self._store is None:
            return self
        field = fields[0]
        reverse = field.startswith("-")
        key = field.lstrip("-+")
        ordenados = sorted(
            self._items,
            key=lambda doc: (getattr(doc, key, None) is None, getattr(doc, key, None)),
            reverse=reverse,
        )
        return FakeQuerySet(ordenados, self._store)

    def delete(self):
        borrados = len(self._items)
        if self._store is not None:
            for item in self._items:
                if item in self._store:
                    self._store.remove(item)
        return borrados

    def update(self, **kwargs):
        for item in self._items:
            for key, value in kwargs.items():
                setattr(item, key.replace("set__", ""), value)
        return len(self._items)


class _ObjectsManager:
    """`Documento.objects` es a la vez llamable y portador de `.insert`."""

    def __init__(self, owner):
        self._owner = owner

    def __call__(self, **filters):
        return FakeQuerySet(self._owner._match(filters), self._owner._store)

    def insert(self, docs, **kwargs):
        if self._owner.insert_error:
            raise self._owner.insert_error
        for doc in docs:
            self._owner._store.append(doc)
        return docs

    def __iter__(self):
        return iter(self._owner._store)


class FakeDocumentMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._store = []
        cls.insert_error = None
        cls.save_error = None
        cls.objects = _ObjectsManager(cls)

    # Sufijos de mongoengine que las consultas del pipeline utilizan.
    _OPERATORS = {"in", "nin", "iexact", "icontains", "gte", "lte", "gt", "lt", "ne"}

    def _match(cls, filters):
        def normalize(value):
            # Los campos ReferenceField se comparan por su id.
            if hasattr(value, "id") and not isinstance(value, (str, int)):
                return str(value.id)
            return str(value)

        def matches(doc):
            for raw_key, value in filters.items():
                if raw_key == "__raw__":
                    if not cls._match_raw(doc, value):
                        return False
                    continue

                partes = raw_key.split("__")
                operador = partes[-1] if partes[-1] in cls._OPERATORS else None
                campo = partes[0]
                actual = normalize(getattr(doc, campo, None))

                if operador == "in":
                    if actual not in {normalize(v) for v in value}:
                        return False
                elif operador == "nin":
                    if actual in {normalize(v) for v in value}:
                        return False
                elif operador == "ne":
                    if actual == normalize(value):
                        return False
                elif operador == "iexact":
                    if actual.lower() != str(value).lower():
                        return False
                elif operador == "icontains":
                    if str(value).lower() not in actual.lower():
                        return False
                elif actual != normalize(value):
                    return False
            return True

        return [doc for doc in cls._store if matches(doc)]

    def _match_raw(cls, doc, raw):
        if "$or" in raw:
            return any(
                all(getattr(doc, k, None) == v for k, v in clause.items())
                for clause in raw["$or"]
            )
        return all(getattr(doc, k, None) == v for k, v in raw.items())

    def reset(cls, docs=None):
        # Se vacía la lista en el sitio (no se reemplaza) para que los managers
        # personalizados que capturaron `_store` sigan viendo el mismo objeto.
        cls._store.clear()
        cls._store.extend(docs or [])
        cls.insert_error = None
        cls.save_error = None
        return cls


class FakeDocument(metaclass=FakeDocumentMeta):
    """Documento mínimo: atributos libres y `save()` que persiste en el store."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.setdefault("id", f"auto-{id(self)}")

    def save(self):
        if type(self).save_error:
            raise type(self).save_error
        if self not in type(self)._store:
            type(self)._store.append(self)
        return self

    def reload(self):
        return self

    def to_mongo(self):
        return dict(self.__dict__)

    def __repr__(self):
        return f"<{type(self).__name__} {self.__dict__}>"


def make_document(name: str, docs: List[Any] = None):
    """Crea una clase de documento independiente con su propio store."""
    cls = FakeDocumentMeta(name, (FakeDocument,), {})
    cls.reset(docs)
    return cls


class Ref:
    """Sustituto de un ReferenceField: solo expone `.id`."""

    def __init__(self, id_value):
        self.id = id_value

    def __eq__(self, other):
        return str(self.id) == str(getattr(other, "id", other))

    def __hash__(self):
        return hash(str(self.id))

    def __str__(self):
        return str(self.id)


class Deforestation:
    """Sub-documento embebido con hectáreas y proporción."""

    def __init__(self, ha=0.0, prop=0.0):
        self.ha = ha
        self.prop = prop


def oid(value: str = None) -> str:
    """ObjectId de mentira: conserva el valor como string."""
    return str(value) if value is not None else "oid-generado"


VALID_OID = "507f1f77bcf86cd799439011"


def make_enum(values: Dict[str, str]):
    """Enum mínimo compatible con `Enum(valor)` y `Enum.NOMBRE.value`."""
    from enum import Enum

    return Enum("FakeEnum", values)
