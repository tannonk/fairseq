## Review Response Generation for Online Hospitality Reviews

This directory contains the model, task, architecture and
supporting scripts for review response generation for the hospitality domain. This
bulk of this work was done as part of the ReAdvisor project
at the University of Zurich 2020/2021. 

This implementation is based on the paper 'Automating App
Review Response Generation' by Gao et al. 2019
[https://ieeexplore.ieee.org/document/8952476].

There are some slight differences to the implementation by
Gao et al.:

1. We use LSTMs instead of GRUs
2. We do not make use of the keyword component since it is
   shown to bring little gain in performance in the original
   authors' ablation study.
3. We do not implement the explicit review length component (yet)
4. We use a custom sentiment engine which provides
   aspect-level sentiment analysis on input reviews in the
   form of a 25-d vector (a flattened 5x5 matrix).


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

---

Author: Tannon Kew

Email: kew@cl.uzh.ch