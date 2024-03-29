import os
import shutil
import sys
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from onnxrt_backend_dev import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "onnxrt-backend-dev"
copyright = "2024"
author = "Microsoft"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]
html_sourcelink_suffix = ""

issues_github_path = "microsoft/onnxrt-backend-dev"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "onnxrt-backend-dev",
    (
        "https://github.com/microsoft/onnxrt-backend-dev/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sklearn-onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
]

nitpick_ignore_regex = [
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
    ("py:class", ".*onnxruntime[.].*"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

epkg_dictionary = {
    "CatBoost": "https://catboost.ai/",
    "cProfile.Profile": "https://docs.python.org/3/library/profile.html#profile.Profile",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "JSON": "https://en.wikipedia.org/wiki/JSON",
    "onnx": "https://onnx.ai/onnx/",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "ONNX Zoo": "https://github.com/onnx/models",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "numpy": "https://numpy.org/",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "torch.onnx": "https://pytorch.org/docs/stable/onnx.html",
}


def setup(app):
    this = os.path.dirname(__file__)
    to_copy = [
        os.path.join(this, "..", "CHANGELOGS.rst"),
        os.path.join(this, "..", "LICENSE"),
    ]
    for name in to_copy:
        shutil.copy(name, this)
