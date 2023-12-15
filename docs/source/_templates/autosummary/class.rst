{% extends "!autosummary/class.rst" %}

{% set methods =(methods| reject("equalto", "__init__") |list) %}

{% block methods %}

{% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}

{% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}
