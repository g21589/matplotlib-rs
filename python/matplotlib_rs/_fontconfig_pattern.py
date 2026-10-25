# Vendored from matplotlib 3.11 (see licenses/LICENSE_MATPLOTLIB), with the
# pyparsing grammar replaced by an equivalent hand-written parser.
"""
A module for parsing and generating `fontconfig patterns`_.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
"""

from functools import lru_cache, partial
import re


_family_punc = r'\\\-:,'
_family_unescape = partial(re.compile(r'\\(?=[%s])' % _family_punc).sub, '')
_family_escape = partial(re.compile(r'(?=[%s])' % _family_punc).sub, r'\\')
_value_punc = r'\\=_:,'
_value_unescape = partial(re.compile(r'\\(?=[%s])' % _value_punc).sub, '')
_value_escape = partial(re.compile(r'(?=[%s])' % _value_punc).sub, r'\\')


_CONSTANTS = {
    'thin':           ('weight', 'light'),
    'extralight':     ('weight', 'light'),
    'ultralight':     ('weight', 'light'),
    'light':          ('weight', 'light'),
    'book':           ('weight', 'book'),
    'regular':        ('weight', 'regular'),
    'normal':         ('weight', 'normal'),
    'medium':         ('weight', 'medium'),
    'demibold':       ('weight', 'demibold'),
    'semibold':       ('weight', 'semibold'),
    'bold':           ('weight', 'bold'),
    'extrabold':      ('weight', 'extra bold'),
    'black':          ('weight', 'black'),
    'heavy':          ('weight', 'heavy'),
    'roman':          ('slant', 'normal'),
    'italic':         ('slant', 'italic'),
    'oblique':        ('slant', 'oblique'),
    'ultracondensed': ('width', 'ultra-condensed'),
    'extracondensed': ('width', 'extra-condensed'),
    'condensed':      ('width', 'condensed'),
    'semicondensed':  ('width', 'semi-condensed'),
    'expanded':       ('width', 'expanded'),
    'extraexpanded':  ('width', 'extra-expanded'),
    'ultraexpanded':  ('width', 'ultra-expanded'),
}



_FAMILY_RE = re.compile(fr"([^{_family_punc}]|(\\[{_family_punc}]))*")
_SIZE_RE = re.compile(r"([0-9]+\.?[0-9]*|\.[0-9]+)")
_NAME_RE = re.compile(r"[a-z]+")
_VALUE_RE = re.compile(fr"([^{_value_punc}]|(\\[{_value_punc}]))*")
_CONST_RE = re.compile("|".join(sorted(map(re.escape, _CONSTANTS), key=len, reverse=True)))


def _parse(pattern):
    pos = 0
    n = len(pattern)
    families = sizes = None
    props = []

    def comma(regex, pos):
        m = regex.match(pattern, pos)
        if m is None:
            return None, pos
        out = [m.group()]
        pos = m.end()
        while pos < n and pattern[pos] == ",":
            m = regex.match(pattern, pos + 1)
            if m is None:
                break
            out.append(m.group())
            pos = m.end()
        return out, pos

    families, pos = comma(_FAMILY_RE, pos)
    if pos < n and pattern[pos] == "-":
        sizes, p2 = comma(_SIZE_RE, pos + 1)
        if sizes is None:
            raise ValueError(f"Invalid fontconfig pattern {pattern!r}")
        pos = p2
    while pos < n and pattern[pos] == ":":
        pos += 1
        m = _NAME_RE.match(pattern, pos)
        if m and m.end() < n and pattern[m.end()] == "=":
            vals, pos = comma(_VALUE_RE, m.end() + 1)
            props.append([m.group(), *vals])
            continue
        m = _CONST_RE.match(pattern, pos)
        if m is None:
            raise ValueError(f"Invalid fontconfig pattern {pattern!r}")
        props.append([m.group()])
        pos = m.end()
    if pos != n:
        raise ValueError(f"Invalid fontconfig pattern {pattern!r}")
    return families, sizes, props


@lru_cache
def parse_fontconfig_pattern(pattern):
    """
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    """
    families, sizes, properties = _parse(pattern)
    props = {}
    if families is not None:
        props["family"] = [*map(_family_unescape, families)]
    if sizes is not None:
        props["size"] = [*sizes]
    for prop in properties:
        if len(prop) == 1:
            prop = _CONSTANTS[prop[0]]
        k, *v = prop
        props.setdefault(k, []).extend(map(_value_unescape, v))
    return props


def generate_fontconfig_pattern(d):
    """Convert a `.FontProperties` to a fontconfig pattern string."""
    kvs = [(k, getattr(d, f"get_{k}")())
           for k in ["style", "variant", "weight", "stretch", "file", "size"]]
    # Families is given first without a leading keyword.  Other entries (which
    # are necessarily scalar) are given as key=value, skipping Nones.
    return (",".join(_family_escape(f) for f in d.get_family())
            + "".join(f":{k}={_value_escape(str(v))}"
                      for k, v in kvs if v is not None))
