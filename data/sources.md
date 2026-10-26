# Data sources

All inputs used in this study are **public surveillance series** published by
official agencies and associated reports. No individual-level or unpublished data
were used. Every plotted or tabulated number in the paper is produced directly
from these series; the deposited scripts transcribe and archive them.

## Primary surveillance agencies (weekly case / death series)

- **UK Health Security Agency (UKHSA)** — "Outbreaks under monitoring" reports,
  wk21–wk41 at DRC data dates; the verified weekly table (`data/weekly_series.csv`)
  is transcribed from these.
- **World Health Organization (WHO)** — DRC Ebola situation reports and bulletins.
- **European Centre for Disease Prevention and Control (ECDC)** — "hospitalized in
  isolation" series (3, 10, 12, 14 October 2026); read as an upper-bound proxy for
  the hospitalized stock.
- **Africa CDC** — continental outbreak summaries.
- **DRC Ministry of Public Health (Ministère de la Santé)** — national declarations
  and data dates; 21 May 2026 outbreak declaration anchor.

## Contextual and methodological references (cited in the paper)

1. SSHAP (2026). *Contextual Note on the Ebola Bundibugyo Outbreak in Ituri (2026).*
   Social Science in Humanitarian Action Platform.
   https://www.socialscienceinaction.org/resources/contextual-note-on-the-ebola-bundibugyo-outbreak-in-ituri-2026

2. Kheth'Impilo-DRC (2026). *Emergency Ebola outbreak response, Congo DR: project
   report, 1 July–31 August 2026* (Béni health zone, North Kivu).
   https://cl.globalgiving.org/projects/emergency-ebola-outbreak-response-congo-dr/reports

3. de Padua B, Akhmetzhanov AR (2026). Estimated transmissibility and case
   fatality rate of Bundibugyo virus, Uganda, 2007. *Emerg Infect Dis.*
   32(10):1762–1764. https://wwwnc.cdc.gov/eid/article/32/10/26-1175

4. WHO (2020). 10th Ebola virus disease outbreak in the DRC declared over.
   https://www.who.int/news/item/25-06-2020-10th-ebola-outbreak-in-the-democratic-republic-of-the-congo-declared-over

Additional modelling and parameter references (incubation period, serial interval,
generation interval, nosocomial reduction factor, transmission heterogeneity) are
enumerated in the manuscript bibliography (LaTeX `thebibliography` environment) and
its Supplementary Material.

## Reproducibility note

The deposition is completed **before submission**; the Zenodo archived-snapshot DOI
is issued after acceptance and is inserted at that time (placeholder
`10.5281/zenodo.XXXXXXX` in the current files). The repository URL and the Zenodo
DOI are referenced in the manuscript's Code and data availability section and in
the PLOS submission package (prepared separately from this code repository).
