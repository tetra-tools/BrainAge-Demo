#!/usr/bin/env Rscript
# Demographics tables of the thesis (chapter IV): subject level and scan level, by split.
#
# Usage (from the repository root):
#   Rscript analysis/demographics_tables.R [--pred-dir DIR] [--out-dir DIR] [--model-tag TAG]
#                                          [--terms FILE] [--no-tables] [--no-strict]
#   defaults: --pred-dir data/predictions  --out-dir results/analysis/demographics  (relative to the repo root)
#
# Reads the prediction tables of ONE model (predictions_<split>_<tag>.tsv and
# predictions_prediction-<split>_<tag>.tsv), keeps one row per scan (native original scan) and writes:
#   demographics_table_split.docx        subject level, by split    (needs gtsummary + gt)
#   demographics_table_split_scans.docx  scan level, by split, and augmented samples by transform as
#                                        n (% of the split's augmented samples)   (needs gtsummary + gt)
#   augmentation_counts.tsv              augmented samples per split and transform (n, %)
#   subject_level.tsv, scan_level.tsv    the data behind the two tables
#   split_counts.tsv                     scans, participants and CN participants (healthy == 1) per split
#   diagnosis_flag_counts.tsv            native scans per diagnosis flag and split, counted as the
#                                        Figure 13 forest plot (n_positive), and how many are healthy
#   diagnosis_group_counts.tsv           native scans and participants per proposed diagnosis group
#                                        (analysis/diagnosis_terms.csv) and split, healthy scans, the
#                                        Other group a split merges it into (< 30 scans), and the
#                                        patients without any group (Unclassified_patient)
#   demographics_checks.txt              counts and label checks
#
# Splits: train = Training; test = Test (hyperparameter selection);
#         validation = External validation (camcan, oasis3; main result).
# Two diagnosis classifications:
#  - Figure 13 flags (neuro_flag_*, psych_flag_*): the rules of the notebook (analysis/diagnosis_groups.py),
#    kept to cross-check the Figure 13 labels (diagnosis_flag_counts.tsv);
#  - proposed groups (group_*): analysis/diagnosis_terms.csv with the rules of
#    analysis/diagnosis_classification.py; these are the "Diagnosis group" rows of the scan table.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(purrr)
  library(tidyr)
})

DEFAULT_MODEL_TAG <- "lossmae_h32_lr0.04_batch16_dr0.4_model_best_score"
APOE_MISSING_TEXT <- c("", "na", "unknown", "unknow")   # with NA and the numeric code -1 (-1, -1.0)
SPLIT_LABELS <- c(train = "Training", test = "Test (hyperparameter selection)",
                  validation = "External validation (Cam-CAN, OASIS-3)")
EXTERNAL_DATABASES <- c("camcan", "oasis3")
AUGMENTATION_TRANSFORMS <- c("Flip", "Blur", "Noise", "Motion", "Ghosting", "Spike")
# Per-file columns ignored when comparing rows of the two files (as analysis_io.DROP_COLS,
# keeping the augmentation columns so different augmented samples stay distinct)
FILE_COLUMNS <- c("subject_image_count", "subject_sample_count", "image_augmentation", "apoe", "split",
                  "t1_path", "infile_index", "path", "ptfile_index", "datasetfile")

# ---------------------------------------------------------------- diagnosis rules (as the notebook)
NEURO_EXCLUDE <- tolower(c("Undetermined", "NA", "Other", "Other Cognitive Disorder", "Unknown",
                           "Alcohol-related", "Medical Illness-related", "CNS Neoplasm", "Epilepsy",
                           "Substance-related", "Posterior Cortical Atrophy", "Essential Tremor"))
PSYCH_EXCLUDE <- tolower(c("Undetermined", "NA", "Other", "Unknown", "surgery"))
NEURO_MUTATION <- c("LRRK2", "GBA", "SNCA", "PRKN", "SWEDD")
NEURO_PRODROMAL <- c("Prodromal Synucleinopathy")
NEURO_PATTERNS <- c(
  Impaired_not_MCI = "^(?:Impaired not MCI)$",
  MCI = "^(?:MCI|Primary Progressive Aphasia)$",
  Vascular = "^(?:Vascular Brain Injury|PostStroke|Vascular Dementia|Stroke)$",
  TBI = "^(?:Traumatic Brain Injury)$",
  FTD = "^(?:FrontoTemporal Dementia)$",
  PSP = "^(?:Progressive Supranuclear Palsy)$",
  LBD = "^(?:Dementia with Lewy Bodies)$",
  CBS = "^(?:CorticoBasal Syndrome)$",
  NPH = "^(?:Normal Pressure Hydrocephalus)$",
  Parkinson = "^(?:Parkinson's Disease|Parkinsons Disease)$",
  SCC = "^(?:Subjective Cognitive Complaint|Subjective Cognitive Concern)$"
)
PSYCH_PATTERNS <- c(
  Bipolar_and_Dysthymia = "^(?:Bipolar Disorder|Dysthymia)$",
  Schizophrenia = "^(?:Schizophrenia)$",
  PTSD = "^(?:Post-Traumatic Stress Disorder|PostTraumatic Stress Disorder)$",
  MDD = "^(?:Major Depressive Disorder)$",
  Anxiety = "^(?:Anxiety Disorder|Generalized Anxiety Disorder|Panic Disorder|GAD)$",
  OCD = "^(?:ObsessiveCompulsive Disorder|Obsessive Compulsive Disorder|OCD)$",
  Pain = "^(?:Pain|Osteoarthritis|Fibromyalgia|Primary Tremor|surgery)$",
  Developmental = paste0("^(?:Attention Deficit Hyperactivity Disorder|Autistic Spectrum Disorders|",
                         "Developmental Neuropsychiatric Disorder|Autism Spectrum Disorder|Autism|ASD|ADHD)$"),
  Addiction = "^(?:Alcohol Use Disorder|Substance Use Disorder|Cannabis)$"
)

dx_tokens <- function(dx) {
  # notebook: replace ; and | by , ; strip spaces around commas and at both ends; drop empty tokens
  if (is.na(dx) || dx == "") return(character(0))
  s <- str_replace_all(dx, "[;|]", ",")
  s <- str_replace_all(str_trim(s), "\\s*,\\s*", ",")
  toks <- str_split(s, fixed(","))[[1]]
  toks[toks != ""]
}

detect_ad <- function(dx) {
  if (is.na(dx) || dx == "") return(FALSE)
  low <- tolower(dx)
  if (str_detect(low, fixed("alzheimer"))) return(TRUE)
  str_detect(low, fixed("dementia")) &&
    !any(str_detect(low, fixed(c("vascular", "frontotemporal", "lewy", "parkinson", "corticobasal",
                                 "progressive supranuclear", "hydrocephalus"))))
}

dx_groups <- function(dx, domain) {
  # groups set for one raw string (same as build_multilabel_frame; CN excluded)
  toks <- dx_tokens(dx)
  exclude <- if (domain == "neuro") NEURO_EXCLUDE else PSYCH_EXCLUDE
  if (length(toks) == 0 || all(tolower(toks) %in% exclude)) return(character(0))
  kept <- toks[!tolower(toks) %in% exclude]
  patterns <- if (domain == "neuro") NEURO_PATTERNS else PSYCH_PATTERNS
  groups <- names(patterns)[vapply(patterns, function(p)
    any(str_detect(kept, regex(p, ignore_case = TRUE))), logical(1))]
  if (domain == "neuro") {
    if (detect_ad(dx)) groups <- c(groups, "AD")
    has <- function(words) any(vapply(words, function(w) any(str_detect(tolower(kept), fixed(tolower(w)))),
                                      logical(1)))
    if (has(NEURO_MUTATION)) groups <- c(groups, "Mutation_PD")
    if (has(NEURO_PRODROMAL)) groups <- c(groups, "Prodromal_PD")
  }
  groups
}

# ---------------------------------------------------------------- proposed classification
# Same rules as analysis/diagnosis_classification.py (R1-R10), from the same table.
TERM_TYPES <- c("CN", "GROUP", "OTHER_DISEASE", "SYMPTOM", "MOVEMENT_SYNDROME", "PPA", "DEMENTIA",
                "OTHER_COGNITIVE", "ATTRIBUTION", "DESCRIPTOR", "MISSING")
CAUSES_OF_DEMENTIA <- c("AD", "Vascular", "FTD", "Atypical_parkinsonism", "Parkinson", "NPH", "TBI")
EXPLAINS_PARKINSONISM <- c("Parkinson", "Mutation_PD", "Prodromal_PD", "Atypical_parkinsonism")
OTHER_DISEASE <- c(neurological = "Other_neuro_disease", psychiatric = "Other_psych_disease")
ALWAYS_KEPT <- c("Other_neuro_disease", "Other_psych_disease", "Other_neuro_symptom", "Other_psych_symptom")
UNCLASSIFIED <- "Unclassified_patient"
MIN_SCANS <- 30

load_terms <- function(path) {
  terms <- read_csv(path, col_types = cols(.default = col_character()), na = character(), progress = FALSE,
                    locale = locale(encoding = "UTF-8"))
  terms$key <- tolower(str_trim(terms$term))
  bad <- setdiff(unique(terms$type), TERM_TYPES)
  if (length(bad)) stop(basename(path), ": unknown type(s) ", paste(bad, collapse = ", "), call. = FALSE)
  if (anyDuplicated(terms$key)) stop(basename(path), ": terms listed twice: ",
                                     paste(unique(terms$key[duplicated(terms$key)]), collapse = ", "), call. = FALSE)
  terms
}

split_terms <- function(dx) {
  if (is.na(dx) || dx == "") return(character(0))
  toks <- str_trim(str_split(str_replace_all(dx, "[;|]", ","), fixed(","))[[1]])
  toks[toks != ""]
}

classify_cell <- function(dx, terms) {
  toks <- split_terms(dx)
  idx <- match(tolower(toks), terms$key)
  typ <- ifelse(is.na(idx), "UNMAPPED", terms$type[idx])
  grp <- terms$group[idx]
  dom <- terms$domain[idx]
  disease <- typ %in% c("GROUP", "OTHER_DISEASE")
  neuro <- unique(grp[disease & dom %in% "neurological"])
  psych <- unique(grp[disease & dom %in% "psychiatric"])
  if ("DEMENTIA" %in% typ && !any(neuro %in% CAUSES_OF_DEMENTIA)) {               # R6
    neuro <- c(neuro, if (any(typ %in% c("ATTRIBUTION", "OTHER_COGNITIVE"))) OTHER_DISEASE[["neurological"]]
                      else "Dementia_unspecified")
  }
  if (any(typ %in% c("OTHER_COGNITIVE", "ATTRIBUTION")) && !length(neuro)) neuro <- OTHER_DISEASE[["neurological"]]
  if ("MOVEMENT_SYNDROME" %in% typ && !any(neuro %in% EXPLAINS_PARKINSONISM)) neuro <- c(neuro, "Movement_disorder")
  if ("PPA" %in% typ && !length(neuro)) neuro <- "MCI"                              # R7
  if (!length(neuro)) neuro <- unique(grp[typ == "SYMPTOM" & dom %in% "neurological"])   # R5
  if (!length(psych)) psych <- unique(grp[typ == "SYMPTOM" & dom %in% "psychiatric"])
  list(groups = unique(c(neuro, psych)), unmapped = toks[is.na(idx)])
}

all_groups <- function(terms) {
  g <- terms[terms$group != "" & terms$type != "CN", ]
  c(unique(g$group[g$domain == "neurological"]), unique(g$group[g$domain == "psychiatric"]))
}

group_domain <- function(group, terms) {
  vapply(group, function(x) {
    if (x == UNCLASSIFIED) return("both")
    d <- terms$domain[terms$group == x & terms$domain != ""]
    if (length(d)) d[1] else ""
  }, character(1), USE.NAMES = FALSE)
}

add_groups <- function(scans, terms) {
  # one logical column per proposed group (both diagnosis columns of a scan together), and
  # Unclassified_patient for patients (healthy == 0) left without any group
  pairs <- distinct(scans, neurological_diagnosis, psychiatric_diagnosis)
  res <- map2(pairs$neurological_diagnosis, pairs$psychiatric_diagnosis, function(n, p) {
    a <- classify_cell(n, terms)
    b <- classify_cell(p, terms)
    list(groups = unique(c(a$groups, b$groups)), unmapped = c(a$unmapped, b$unmapped))
  })
  pairs$groups <- map(res, "groups")
  pairs$unmapped <- map_chr(res, ~ paste(unique(.x$unmapped), collapse = "; "))
  scans <- left_join(scans, pairs, by = c("neurological_diagnosis", "psychiatric_diagnosis"))
  for (g in all_groups(terms)) scans[[paste0("group_", g)]] <- vapply(scans$groups, function(x) g %in% x, logical(1))
  scans[[paste0("group_", UNCLASSIFIED)]] <- lengths(scans$groups) == 0 & scans$healthy %in% 0
  scans %>% rename(unmapped_terms = unmapped) %>% select(-groups)
}

group_counts <- function(scans, terms) {
  long <- scans %>%
    select(split, database, subject_id, healthy, starts_with("group_")) %>%
    pivot_longer(starts_with("group_"), names_to = "group", values_to = "value") %>%
    filter(value) %>%
    mutate(group = str_remove(group, "^group_"))
  if (!nrow(long)) return(tibble(group = character(0)))
  counts <- long %>% group_by(group, split) %>%
    summarise(scans = n(), participants = n_distinct(paste(database, subject_id)),
              healthy = sum(healthy == 1), .groups = "drop")
  merged <- counts %>%
    mutate(domain = group_domain(group, terms),
           merged_into = if_else(scans < MIN_SCANS & !group %in% c(ALWAYS_KEPT, UNCLASSIFIED) &
                                   domain %in% names(OTHER_DISEASE), unname(OTHER_DISEASE[domain]), "")) %>%
    select(group, split, merged_into)
  out <- counts %>% left_join(merged, by = c("group", "split")) %>%
    pivot_wider(names_from = split, values_from = c(scans, participants, healthy, merged_into),
                names_glue = "{.value}_{split}")
  order <- c(all_groups(terms), UNCLASSIFIED)
  out %>% mutate(domain = group_domain(group, terms), .after = group) %>%
    arrange(match(group, order))
}

add_flags <- function(scans) {
  neuro <- lapply(scans$neurological_diagnosis, dx_groups, domain = "neuro")
  psych <- lapply(scans$psychiatric_diagnosis, dx_groups, domain = "psych")
  for (g in c(names(NEURO_PATTERNS), "AD", "Mutation_PD", "Prodromal_PD")) {
    scans[[paste0("neuro_flag_", g)]] <- vapply(neuro, function(x) g %in% x, logical(1))
  }
  for (g in names(PSYCH_PATTERNS)) {
    scans[[paste0("psych_flag_", g)]] <- vapply(psych, function(x) g %in% x, logical(1))
  }
  scans
}

# ---------------------------------------------------------------- reading and checks
read_predictions <- function(pred_dir, model_tag) {
  suffix <- paste0("_", model_tag, ".tsv")
  files <- list.files(pred_dir, full.names = TRUE)
  files <- files[startsWith(basename(files), "predictions_") & endsWith(basename(files), suffix)]
  if (length(files) == 0) stop("no prediction table ending in '", suffix, "' in ", pred_dir, call. = FALSE)
  map_dfr(files, function(f) {
    # every column as text: no type guessing, IDs keep leading zeros, nothing silently becomes NA
    d <- read_tsv(f, col_types = cols(.default = col_character()), progress = FALSE,
                  locale = locale(encoding = "UTF-8")) %>%
      select(-starts_with("..."))
    text <- unlist(d, use.names = FALSE)
    if (any(!validUTF8(text[!is.na(text)]))) stop(basename(f), " is not UTF-8 encoded; re-save it as UTF-8.", call. = FALSE)
    mutate(d, datasetfile = str_remove(str_remove(basename(f), "^predictions_"), fixed(suffix)))
  })
}

code_apoe <- function(x) {
  # 0 = non-carrier, > 0 = carrier; -1, -1.0, "Unknown", empty and NA are missing. Anything else is
  # not recognised: returned as NA so prepare() can report it, and shown as "Unknown".
  v <- str_trim(as.character(x))
  num <- suppressWarnings(as.numeric(v))
  case_when(is.na(v) | tolower(v) %in% APOE_MISSING_TEXT | num %in% -1 ~ "Unknown",
            num %in% 0 ~ "Non-carrier",
            !is.na(num) & num > 0 ~ "Carrier",
            TRUE ~ NA_character_)
}

as_flag <- function(x) {
  v <- tolower(str_trim(x))
  case_when(v %in% c("true", "1", "1.0") ~ TRUE, v %in% c("false", "0", "0.0") ~ FALSE, TRUE ~ NA)
}

prepare <- function(raw, terms) {
  problems <- character(0)
  warnings <- character(0)
  required <- c("subject_id", "session_id", "database", "split", "augmentation", "is_original", "age",
                "predicted_age_years", "sex", "healthy", "neurological_diagnosis", "psychiatric_diagnosis")
  missing_cols <- setdiff(required, names(raw))
  if (length(missing_cols)) stop("missing column(s): ", paste(missing_cols, collapse = ", "), call. = FALSE)
  if (!"apoe" %in% names(raw)) raw$apoe <- NA_character_

  show <- function(x, n = 8) paste(head(unique(x), n), collapse = ", ")
  for (col in c("age", "predicted_age_years")) {
    v <- suppressWarnings(as.numeric(raw[[col]]))
    bad <- raw[[col]][!is.na(raw[[col]]) & is.na(v)]
    if (length(bad)) problems <- c(problems, sprintf("'%s': %d values are not numbers: %s", col, length(bad), show(bad)))
    raw[[col]] <- v
  }
  raw$age <- floor(raw$age)  # chronological age floored to whole years in every analysis (design)
  raw$is_original_flag <- as_flag(raw$is_original)
  raw$healthy_num <- suppressWarnings(as.numeric(raw$healthy))

  aug <- raw$augmentation
  near <- !is.na(aug) & tolower(str_trim(aug)) == "native" & aug != "native"
  if (any(near)) problems <- c(problems, sprintf("'augmentation': %d rows labelled %s instead of exactly 'native'",
                                                 sum(near), show(aug[near])))
  empty_orig <- (is.na(aug) | str_trim(aug) == "") & raw$is_original_flag %in% TRUE
  if (any(empty_orig)) problems <- c(problems, sprintf("'augmentation': %d original rows have no label", sum(empty_orig)))
  native_like <- is.na(aug) | tolower(str_trim(aug)) %in% c("native", "")
  bad_orig <- native_like & is.na(raw$is_original_flag)
  if (any(bad_orig)) problems <- c(problems, sprintf("'is_original': %d native-like rows are not TRUE/FALSE: %s",
                                                     sum(bad_orig), show(raw$is_original[bad_orig])))

  file_split <- str_remove(raw$datasetfile, "^prediction-")
  bad_split <- is.na(raw$split) | raw$split != file_split | !raw$split %in% names(SPLIT_LABELS)
  if (any(bad_split)) problems <- c(problems, sprintf("'split': %d rows disagree with their file name or are not %s: %s",
                                                      sum(bad_split), paste(names(SPLIT_LABELS), collapse = "/"),
                                                      show(paste(raw$split[bad_split], raw$datasetfile[bad_split]))))

  # Rows of a prediction-<split> file that are copies of rows of the <split> (healthy) file are
  # counted once. Repeated rows within one file are kept: two flipped copies of the same scan are
  # identical (same augmentation text, same image, same prediction) but are two samples.
  row_key <- do.call(paste, c(raw[c("split", setdiff(names(raw), FILE_COLUMNS))], sep = "\x1f"))
  from_prediction_file <- startsWith(raw$datasetfile, "prediction-")
  copy_of_healthy_row <- from_prediction_file & row_key %in% row_key[!from_prediction_file]
  raw <- raw %>% mutate(scan_key = paste(split, database, subject_id, session_id, sep = "/"))
  rows <- raw[!copy_of_healthy_row, ]
  native <- rows %>% filter(augmentation == "native", is_original_flag %in% TRUE)
  lost <- setdiff(unique(raw$scan_key), unique(native$scan_key))
  if (length(lost)) problems <- c(problems, sprintf("%d scans have no native original row and would be dropped: %s",
                                                    length(lost), show(lost)))

  # forest-plot rule for native scans: identical native rows (within or across files) count once
  native <- native[!duplicated(native[c("split", setdiff(names(native), c(FILE_COLUMNS, "scan_key")))]), ]
  conflict <- native %>% group_by(scan_key) %>%
    summarise(n = n(), n_values = n_distinct(age, healthy_num, sex), .groups = "drop") %>% filter(n_values > 1)
  if (nrow(conflict)) problems <- c(problems, sprintf("%d scans have native rows with different age/healthy/sex: %s",
                                                      nrow(conflict), show(conflict$scan_key)))
  n_dup <- sum(duplicated(native$scan_key))
  if (n_dup) warnings <- c(warnings, sprintf("%d scans have several native rows with the same age/healthy/sex; one kept",
                                             n_dup))
  native <- native %>% distinct(scan_key, .keep_all = TRUE)
  if (any(is.na(native$age))) warnings <- c(warnings, sprintf("'age': %d native scans have no age", sum(is.na(native$age))))

  bad_h <- !native$healthy_num %in% c(0, 1)
  if (any(bad_h)) problems <- c(problems, sprintf("'healthy': %d native scans are not 0/1: %s",
                                                  sum(bad_h), show(native$healthy[bad_h])))
  ext_bad <- native$split == "validation" & !native$database %in% EXTERNAL_DATABASES
  if (any(ext_bad)) problems <- c(problems, sprintf("'database': %d external validation scans labelled %s",
                                                    sum(ext_bad), show(native$database[ext_bad])))

  # sex: Female/female/Male/male (any case, spaces) are coded; a missing value is filled from the same
  # participant's other native scans when those agree; disagreements are reported, never filled
  native <- native %>%
    mutate(sex_raw = sex,
           sex_norm = tolower(str_trim(sex)),
           sex_norm = if_else(sex_norm %in% c("female", "male"), sex_norm, NA_character_)) %>%
    group_by(database, subject_id) %>%
    mutate(sex_known = paste(sort(unique(na.omit(sex_norm))), collapse = "/"),
           sex_filled = is.na(sex_norm) & sex_known %in% c("female", "male"),
           sex_norm = if_else(sex_filled, sex_known, sex_norm)) %>%
    ungroup()
  odd_sex <- native$sex_raw[!is.na(native$sex_raw) & !tolower(str_trim(native$sex_raw)) %in% c("female", "male")]
  if (length(odd_sex)) warnings <- c(warnings, sprintf("'sex': %d native scans with a value other than female/male: %s",
                                                       length(odd_sex), show(odd_sex)))
  if (any(native$sex_filled)) warnings <- c(warnings, sprintf(
    "'sex': %d native scans had no sex and were given the sex of the same participant's other scans",
    sum(native$sex_filled)))
  mixed <- unique(paste(native$database, native$subject_id)[native$sex_known == "female/male"])
  if (length(mixed)) warnings <- c(warnings, sprintf("'sex': %d participants are female on some scans and male on others (not changed): %s",
                                                     length(mixed), show(mixed)))
  n_missing_sex <- sum(is.na(native$sex_norm))
  if (n_missing_sex) warnings <- c(warnings, sprintf("'sex': %d native scans remain without sex (shown as Unknown)", n_missing_sex))
  odd_apoe <- native$apoe[is.na(code_apoe(native$apoe))]
  if (length(odd_apoe)) warnings <- c(warnings, sprintf("'apoe': %d values not recognised (not 0, > 0, -1, Unknown, empty; shown as Unknown): %s",
                                                        length(odd_apoe), show(odd_apoe)))
  n_paths <- if ("path" %in% names(native)) n_distinct(native$path) else NA
  if (!is.na(n_paths) && n_paths != nrow(native)) {
    warnings <- c(warnings, sprintf("'path' has %d distinct values for %d native scans: scans are keyed by split/database/subject_id/session_id, not by path",
                                    n_paths, nrow(native)))
  }

  # augmented samples per scan, by transform
  samples <- rows %>%
    filter(scan_key %in% native$scan_key) %>%
    mutate(augmented = !(augmentation %in% "native" & is_original_flag %in% TRUE))
  for (t in AUGMENTATION_TRANSFORMS) {
    samples[[paste0("aug_", t)]] <- samples$augmented & str_detect(coalesce(samples$augmentation, ""),
                                                                    regex(t, ignore_case = TRUE))
  }
  per_scan <- samples %>% group_by(scan_key) %>%
    summarise(n_samples = n(), n_augmented = sum(augmented),
              across(starts_with("aug_"), sum), .groups = "drop")
  # one row per augmented sample with its transform (one transform per augmented copy; a label
  # naming several, or none, of the known transforms is shown as such)
  hits <- as.matrix(samples[paste0("aug_", AUGMENTATION_TRANSFORMS)])
  n_hits <- rowSums(hits)
  transform <- ifelse(n_hits == 1, AUGMENTATION_TRANSFORMS[max.col(hits, ties.method = "first")],
                      ifelse(n_hits > 1, "Several transforms", "Unrecognised"))
  augmented_samples <- tibble(split = samples$split, transform = transform)[samples$augmented, ]
  if (any(augmented_samples$transform == "Unrecognised")) warnings <- c(warnings, sprintf(
    "'augmentation': %d augmented samples name none of %s: %s", sum(augmented_samples$transform == "Unrecognised"),
    paste(AUGMENTATION_TRANSFORMS, collapse = "/"),
    show(samples$augmentation[samples$augmented][augmented_samples$transform == "Unrecognised"])))

  scans <- native %>%
    mutate(healthy = healthy_num,
           sex = case_when(sex_norm == "female" ~ "Female", sex_norm == "male" ~ "Male", TRUE ~ "Unknown"),
           apoe_raw = apoe,
           apoe = coalesce(code_apoe(apoe), "Unknown")) %>%
    select(-sex_norm, -sex_known) %>%
    left_join(per_scan, by = "scan_key") %>%
    add_flags() %>%
    add_groups(terms)
  unmapped <- unique(unlist(str_split(scans$unmapped_terms[scans$unmapped_terms != ""], "; ")))
  if (length(unmapped)) warnings <- c(warnings, sprintf(
    "diagnosis terms not in diagnosis_terms.csv (no proposed group; add them to the table): %s", show(unmapped, 50)))
  list(scans = scans, augmented_samples = augmented_samples, problems = problems, warnings = warnings,
       rows_read = nrow(raw), rows_distinct = nrow(rows))
}

subject_level <- function(scans) {
  # sessions in age order; sessions in the same (floored) year ordered by the number in session_id
  # ("ses-2" before "ses-10"), as in fig_brainage_followup.py
  scans %>%
    mutate(ses_num = suppressWarnings(as.numeric(str_extract(session_id, "[0-9]+")))) %>%
    arrange(age, ses_num, session_id) %>%
    group_by(split, database, subject_id) %>%
    summarise(
      n_sessions = n_distinct(session_id),
      # computed before `age` is replaced by the baseline age below
      # two versions: single-session subjects as 0, and subjects with >= 2 sessions only (others NA)
      followup_all = if (n_distinct(session_id) > 1 && sum(!is.na(age)) > 1)
        max(age, na.rm = TRUE) - min(age, na.rm = TRUE) else 0,
      followup_multi = if (n_distinct(session_id) > 1 && sum(!is.na(age)) > 1) followup_all else NA_real_,
      healthy_changes = n_distinct(healthy) > 1,
      n_samples = sum(n_samples),
      age = first(age), sex = first(sex), apoe = first(apoe), healthy = first(healthy),
      .groups = "drop")
}

flag_counts <- function(scans) {
  # as Figure 13 (n_positive): every native scan of the split, healthy or not; plus how many are healthy
  scans %>%
    filter(split %in% c("test", "validation")) %>%
    select(split, healthy, starts_with("neuro_flag_"), starts_with("psych_flag_")) %>%
    pivot_longer(-c(split, healthy), names_to = "flag", values_to = "value") %>%
    group_by(flag, split) %>%
    summarise(n_positive = sum(value), healthy = sum(value & healthy == 1), .groups = "drop") %>%
    pivot_wider(names_from = split, values_from = c(n_positive, healthy), names_glue = "{.value}_{split}") %>%
    arrange(desc(rowSums(pick(starts_with("n_positive")), na.rm = TRUE)))
}

# ---------------------------------------------------------------- tables (gtsummary)
split_factor <- function(x) factor(x, levels = names(SPLIT_LABELS), labels = SPLIT_LABELS)

TRANSFORM_LEVELS <- c(AUGMENTATION_TRANSFORMS, "Several transforms", "Unrecognised")

# augmented samples per split and transform: n and % of the split's augmented samples
augmentation_counts <- function(augmented_samples) {
  augmented_samples %>%
    mutate(transform = factor(transform, levels = TRANSFORM_LEVELS)) %>%
    count(split, transform, name = "samples", .drop = FALSE) %>%
    group_by(split) %>%
    mutate(percent = if (sum(samples)) round(100 * samples / sum(samples), 1) else NA_real_) %>%
    ungroup() %>%
    filter(!(transform %in% c("Several transforms", "Unrecognised") & samples == 0)) %>%
    arrange(match(split, names(SPLIT_LABELS)), transform)
}

relabel_missing <- function(tbl, variable_name, text) {
  # label of the missing-value row of one variable (gtsummary has one missing_text for all variables)
  tryCatch(gtsummary::modify_table_body(tbl, ~ dplyr::mutate(.x, label = dplyr::if_else(
    .data$variable == variable_name & .data$row_type == "missing", text, .data$label))),
    error = function(e) tbl)
}

make_tables <- function(subjects, scans, augmented_samples, out_dir) {
  if (!requireNamespace("gtsummary", quietly = TRUE) || !requireNamespace("gt", quietly = TRUE)) {
    message("gtsummary/gt not installed: .docx tables skipped (subject_level.tsv and scan_level.tsv written).")
    return(invisible(FALSE))
  }
  library(gtsummary)
  stat <- list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)")

  subj <- subjects %>%
    transmute(
      split = split_factor(split),
      healthy = factor(healthy, levels = c(0, 1), labels = c("Non-healthy", "Healthy")),
      age, sex = factor(sex, levels = c("Female", "Male", "Unknown")),
      apoe = factor(apoe, levels = c("Non-carrier", "Carrier", "Unknown")),
      n_sessions, n_samples, followup_all, followup_multi)
  # ages and follow-up are whole years (floored): without `type`, gtsummary would show variables with
  # fewer than 10 distinct values as categories instead of mean (SD)
  tbl_summary(
    subj, by = split, missing = "ifany", statistic = stat, digits = all_continuous() ~ 2,
    type = list(c(age, n_sessions, n_samples, followup_all, followup_multi) ~ "continuous"),
    include = c(healthy, age, sex, apoe, n_sessions, n_samples, followup_all, followup_multi),
    label = list(healthy ~ "Status at baseline", age ~ "Age at baseline (years)", sex ~ "Sex", apoe ~ "APOE",
                 n_sessions ~ "Sessions per subject", n_samples ~ "Samples per subject (native + augmented)",
                 followup_all ~ "Follow-up duration, all subjects (years; single session = 0)",
                 followup_multi ~ "Follow-up duration, subjects with >= 2 sessions (years)")) %>%
    # add_overall() first: it recomputes the table without `by` and merges the rows on their labels,
    # so a label changed before it ("Unknown" -> "Single session") makes the merge fail
    add_overall() %>%
    relabel_missing("followup_multi", "Single session (no follow-up)") %>%
    as_gt() %>%
    gt::gtsave(filename = file.path(out_dir, "demographics_table_split.docx"))

  # proposed diagnosis groups present in the data (a scan can be in several groups)
  present <- names(scans)[startsWith(names(scans), "group_")]
  present <- present[vapply(present, function(g) any(scans[[g]]), logical(1))]
  sc <- scans %>%
    select(split, healthy, age, all_of(present)) %>%
    mutate(split = split_factor(split),
           healthy = factor(healthy, levels = c(0, 1), labels = c("Non-healthy", "Healthy"))) %>%
    rename_with(~ paste0("Diagnosis group: ", str_replace_all(str_remove(.x, "^group_"), "_", " ")),
                starts_with("group_"))
  scan_tbl <- tbl_summary(
    sc, by = split, missing = "ifany", statistic = stat, digits = all_continuous() ~ 2,
    type = list(age ~ "continuous", starts_with("Diagnosis group: ") ~ "dichotomous"),
    include = c(healthy, age, starts_with("Diagnosis group: ")),
    label = list(healthy ~ "Status", age ~ "Age at session (years)")) %>%
    add_overall()

  # augmented samples, one row per sample: n (% of the split's augmented samples) by transform
  aug <- augmented_samples %>%
    mutate(split = split_factor(split),
           transform = droplevels(factor(transform, levels = TRANSFORM_LEVELS)))
  aug_tbl <- tbl_summary(
    aug, by = split, statistic = all_categorical() ~ "{n} ({p}%)", include = transform,
    label = list(transform ~ "Transform (n, % of the split's augmented samples)")) %>%
    add_overall()
  same_splits <- setequal(as.character(unique(sc$split)), as.character(unique(aug$split)))
  if (same_splits) {
    tbl_stack(list(scan_tbl, aug_tbl), group_header = c("Native scans", "Augmented samples")) %>%
      as_gt() %>%
      gt::gtsave(filename = file.path(out_dir, "demographics_table_split_scans.docx"))
  } else {
    # a split without augmented samples: the two tables do not have the same columns
    message("Some splits have no augmented samples: augmentation table written separately.")
    scan_tbl %>% as_gt() %>% gt::gtsave(filename = file.path(out_dir, "demographics_table_split_scans.docx"))
    aug_tbl %>% as_gt() %>% gt::gtsave(filename = file.path(out_dir, "demographics_table_split_augmentation.docx"))
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------- main
script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) return(normalizePath(dirname(sub("^--file=", "", file_arg[1]))))
  file.path(getwd(), "analysis")
}

repo_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[1])), "..")))
  getwd()
}

parse_args <- function(args) {
  opts <- list(pred_dir = NULL, out_dir = NULL, model_tag = DEFAULT_MODEL_TAG, terms = NULL, tables = TRUE,
               strict = TRUE)
  i <- 1
  while (i <= length(args)) {
    key <- args[i]
    if (key %in% c("-h", "--help")) {
      cat("Rscript analysis/demographics_tables.R [--pred-dir DIR] [--out-dir DIR] [--model-tag TAG]",
          "[--terms analysis/diagnosis_terms.csv] [--no-tables] [--no-strict]\n")
      quit(status = 0)
    }
    if (key == "--no-tables") { opts$tables <- FALSE; i <- i + 1; next }
    if (key == "--no-strict") { opts$strict <- FALSE; i <- i + 1; next }
    if (i == length(args)) stop("missing value for ", key, call. = FALSE)
    val <- args[i + 1]
    if (key == "--pred-dir") opts$pred_dir <- val
    else if (key == "--out-dir") opts$out_dir <- val
    else if (key == "--model-tag") opts$model_tag <- val
    else if (key == "--terms") opts$terms <- val
    else stop("unknown option ", key, call. = FALSE)
    i <- i + 2
  }
  opts
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  opts <- parse_args(args)
  root <- repo_root()
  pred_dir <- if (is.null(opts$pred_dir)) file.path(root, "data", "predictions") else opts$pred_dir
  out_dir <- if (is.null(opts$out_dir)) file.path(root, "results", "analysis", "demographics") else opts$out_dir
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  terms <- load_terms(if (is.null(opts$terms)) file.path(script_dir(), "diagnosis_terms.csv") else opts$terms)
  prep <- prepare(read_predictions(pred_dir, opts$model_tag), terms)
  if (length(prep$problems) && opts$strict) {
    stop("labelling/encoding problems would drop scans (use --no-strict to only report them):\n  - ",
         paste(prep$problems, collapse = "\n  - "), call. = FALSE)
  }
  scans <- prep$scans
  subjects <- subject_level(scans)
  write_tsv(subjects, file.path(out_dir, "subject_level.tsv"))
  write_tsv(scans %>% select(-scan_key), file.path(out_dir, "scan_level.tsv"))
  counts <- flag_counts(scans)
  write_tsv(counts, file.path(out_dir, "diagnosis_flag_counts.tsv"))
  groups <- group_counts(scans, terms)
  write_tsv(groups, file.path(out_dir, "diagnosis_group_counts.tsv"))
  aug_counts <- augmentation_counts(prep$augmented_samples)
  write_tsv(aug_counts, file.path(out_dir, "augmentation_counts.tsv"))

  # one row per native scan here; participants = distinct database + subject_id
  per_split <- scans %>% group_by(split) %>%
    summarise(scans = n(), participants = n_distinct(paste(database, subject_id)),
              `CN participants (healthy == 1)` = n_distinct(paste(database, subject_id)[healthy == 1]),
              `CN scans (healthy == 1)` = sum(healthy == 1), .groups = "drop")
  write_tsv(per_split, file.path(out_dir, "split_counts.tsv"))
  checks <- c(
    sprintf("Model tag: %s", opts$model_tag),
    sprintf("Rows read: %d; after removing prediction-file copies of healthy-file rows: %d", prep$rows_read, prep$rows_distinct),
    capture.output(print(as.data.frame(per_split))),
    "", "Every raw sex value (native scans) and how it was coded:",
    capture.output(print(as.data.frame(count(scans, split, sex_raw, sex_filled, sex)), row.names = FALSE)),
    "", "Every raw APOE value (native scans) and how it was coded:",
    capture.output(print(as.data.frame(count(scans, apoe_raw, apoe)), row.names = FALSE)), "",
    "Augmented samples by transform (n, % of the split's augmented samples):",
    capture.output(print(as.data.frame(aug_counts), row.names = FALSE)), "",
    "Proposed diagnosis groups (native scans; Other group if < 30 scans in a split):",
    capture.output(print(as.data.frame(groups), row.names = FALSE)), "",
    sprintf("ERROR: %s", prep$problems), sprintf("WARNING: %s", prep$warnings))
  writeLines(checks, file.path(out_dir, "demographics_checks.txt"))
  cat(checks, sep = "\n")
  print(as.data.frame(counts))
  if (opts$tables) make_tables(subjects, scans, prep$augmented_samples, out_dir)
  invisible(0)
}

if (sys.nframe() == 0L) main()
