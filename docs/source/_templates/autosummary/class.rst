{% extends "!autosummary/class.rst" %}

.. autoclass:: {{ objname }}
{% set methods =(methods| reject("equalto", "__init__") |list) %}

{% block methods %}

{% if methods %}
   .. rubric:: Methods
{% for item in methods %}
   .. automethod:: {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}

{% if attributes %}
   .. rubric:: Attributes
{% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
