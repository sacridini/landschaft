# Habitat patterns

Structural classes of habitat pixels (MSPA, SPA), objects by size,
parcellation and boundary contortion. Tutorial:
[habitat patterns](../tutorials/patterns.md).

## MSPA

::: landschaft.mspa

::: landschaft.mspa_stats

### MSPA class codes

`ls.MSPA_CLASSES` maps each code of an MSPA map to its class:

| Code | Class | Code | Class |
|---|---|---|---|
| 17 | Core | 65 | Loop |
| 9 | Islet | 67 | Loop in edge |
| 5 | Perforation | 69 | Loop in perforation |
| 3 | Edge | 33 | Bridge |
| 1 | Branch | 35 | Bridge in edge |
| 0 | Background | 37 | Bridge in perforation |
| 100 | Core-opening | 220 | Border-opening |
| 129 | Missing | +100 | the class inside a core-opening (`internal=True`) |

## SPA

::: landschaft.spa

::: landschaft.SPAResult

## Objects by size

::: landschaft.accounting

::: landschaft.AccountingResult

## Parcels and shape

::: landschaft.parcellation

::: landschaft.ParcellationResult

::: landschaft.contortion
