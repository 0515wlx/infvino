# Intel® Open Source PRM — notice and how infvino uses it

This directory records infvino's use of **public** Intel documentation. No Intel source code is
vendored or linked. infvino contains only an **original software implementation** written from the
publicly documented algorithm, as expressly permitted below.

## Document used (public)

- **Title**: *Intel® Iris® Xe and UHD Graphics Open Source Programmer's Reference Manual — For the
  2020-2021 11th Generation Intel® Xeon®, Core™, Celeron®, Pentium® Gold Processors based on the
  "Tiger Lake" Platform*, **Volume 7: Memory Cache**.
- **Doc Ref**: `IHD-OS-TGL-Vol 7-12.21` (December 2021, Revision 1.0).
- **Public URL** (Intel): <https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/tiger-lake.html>
  (Volume 7 attachment). Mirror index: <https://kiwitree.net/~lina/intel-gfx-docs/prm>.
- Retrieved 2026-10-07.

## License-relevant notice (reproduced **unmodified**, as permitted by clause (a))

> **Notices and Disclaimers**
>
> Intel technologies may require enabled hardware, software or service activation.
> No product or component can be absolutely secure.
> Code names are used by Intel to identify products, technologies, or services that are in
> development and not publicly available. These are not "commercial" names and not intended to
> function as trademarks
> Customer is responsible for safety of the overall system, including compliance with applicable
> safety-related requirements or standards.
> No license (express or implied, by estoppel or otherwise) to any intellectual property rights is
> granted by this document, with the sole exceptions that a) you may publish an unmodified copy and
> b) code included in this document is licensed subject to Zero-Clause BSD open source license
> (0BSD). You may create software implementations based on this document and in compliance with the
> foregoing that are intended to execute on the Intel product(s) referenced in this document. No
> rights are granted to create modifications or derivatives of this document.
> The products described may contain design defects or errors known as errata which may cause the
> product to deviate from published specifications. Current characterized errata are available on
> request.
> You may not use or facilitate the use of this document in connection with any infringement or other
> legal analysis concerning Intel products described herein. You agree to grant Intel a non-exclusive,
> royalty-free license to any patent claim thereafter drafted which includes subject matter disclosed
> herein.
> Intel disclaims all express and implied warranties, including without limitation, the implied
> warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any
> warranty arising from course of performance, course of dealing, or usage in trade.
> Intel may make changes to specifications and product descriptions at any time, without notice.
> Designers must not rely on the absence or characteristics of any features or instructions marked
> "reserved" or "undefined". Intel reserves these for future definition and shall have no responsibility
> whatsoever for conflicts or incompatibilities arising from future changes to them. The information
> here is subject to change without notice. Do not finalize a design with this information.
> © Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel
> Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

*(The text above is an unmodified copy of the document's own notice. This repository does **not**
redistribute the document itself, nor any modified/derivative copy of it.)*

## How infvino complies

| Obligation / permission | How infvino complies |
|---|---|
| "you may publish an unmodified copy" | Only this verbatim notice block is reproduced. |
| "No rights are granted to create modifications or derivatives of this document" | infvino does **not** ship the PRM, excerpts, figures, register tables, or any adapted copy. Only a **short factual summary** of the replacement algorithm is used, with attribution. |
| "You may create software implementations based on this document" | infvino's `L3LineModel` (`include/infvino/L3LineModel.hpp`, `src/L3LineModel.cpp`) is an **original** implementation of the publicly documented **1-bit LRU** (N-bit vector per set; fill selects the first 0-way and flips it; hits set the way's bit; all-ones clears the vector; "allocate on fill"). No Intel code (0BSD or otherwise) was copied. |
| trademarks / endorsements | infvino is an independent project; nothing here implies Intel sponsorship or endorsement. |

**Facts encoded in infvino (from the public Volume 7):** L3 bank = 480 KB = 120 logical ways ×
64 sets × 64 B lines; ≤104 ways tagged for L3$; 1-bit LRU replacement; "Allocate on fill"; way-based
partitioning via `L3ALLOCREG`; sectors of 2 ways. These are used as **parameters/behaviour**, not as
copied text.

See also the top-level [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
