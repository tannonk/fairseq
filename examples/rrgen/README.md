## Review Response Generation for Online Hospitality Reviews

---

Author: Tannon Kew

Email: kew@cl.uzh.ch

---



## Decoding

Instead of modifying (and potentially breaking) Fairseq's
SequenceGenerator found in `sequence_generator.py`, we
replace it with a task-specific SequenceGenerator class
defined in `sequence_generator_rrgen.py`.

SequenceGenerator is called in the rrgen_translation task,
which contains a method `build_generator`, overriding the
default `build_generator` method, which is inherited from
the `fairseq_task` base class.

NB. funtionality is essentially the same in both
`build_generator` methods. Only the import statement changes
for the initialisation of the SequenceGenerator object.

