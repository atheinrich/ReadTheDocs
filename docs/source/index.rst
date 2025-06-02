Primary Title
=============

Typical text:
regular,
*italic*,
**bold**.

Linked text: `website <https://github.com/atheinrich/>`_,
:doc:`page`,
:ref:`section`.

Other text:
   tabbed,
   ``inline``,
   :sub:`subscript`,
   :sup:`superscript`,
      doubly tabbed.


Secondary Title
---------------

.. note::

   Textbox.

.. code-block:: console

   (.env) $ console commands

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']


Secondary Title
---------------

Table of contents: 

.. toctree::

   page

