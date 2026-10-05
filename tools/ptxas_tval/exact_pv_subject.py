"""Fail-closed identity of the PTX subject reviewed against ExactPvExact.v.

This pin binds every instruction, operand, and declaration, rather than just
the occurrence of an opcode. It is a trusted transcription boundary, not an
operational PTX proof. Updating the pin requires reviewing the entire changed
subject against the theorem and rerunning the source/PTX and device gates.
Comments and whitespace are excluded; semantic changes require a new review.
"""
import hashlib
from pathlib import Path
if __package__:
    from . import ptxsource
else:
    import ptxsource


def require_proved_subject(source):
    try:
        text = source.decode('utf-8') if isinstance(source, bytes) else source
        expected = Path(__file__).with_name('exact_pv_subject.sha256').read_text().strip()
    except (UnicodeError, OSError) as error:
        raise ValueError(f'cannot identify reviewed exact_pv proof subject: {error}') from error
    try:
        subject = ' '.join(ptxsource.strip_comments(text).split())
    except Exception as error:
        raise ValueError(f'cannot parse reviewed exact_pv proof subject: {error}') from error
    if hashlib.sha256(subject.encode('utf-8')).hexdigest() != expected:
        raise ValueError('PTX differs from the reviewed ExactPvExact proof subject; '
                         'translation equivalence alone does not prove the PV algorithm')
