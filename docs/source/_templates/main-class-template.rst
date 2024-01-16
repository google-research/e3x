{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members:
   :show-inheritance:

   {% block methods %}
   .. automethod:: __call__
   {% endblock %}

