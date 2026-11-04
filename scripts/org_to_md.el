;;; org_to_md.el --- Markdown layout for org_to_md.sh  -*- lexical-binding: t; -*-

;; Author: Rohit Goswami (HaoZeke)
;; License: MIT

;; ox-md pads every bullet to four columns ("-   ", "1.  ") and indents
;; the rest of the item by four.  It does so in lists, in headlines
;; deeper than the H: option (which it writes as list items) and in the
;; table of contents.  snapper, the semantic line-break checker
;; (https://github.com/TurtleTech-ehf/snapper), wants one space after
;; the bullet.  Rewriting only the bullet would move an indented
;; code block inside an item two columns to the right, so the filters
;; below also shift the item's other lines left by the padding that
;; went, and give the table of contents two columns per level.
;;
;; ox-md also indents code blocks by four columns and writes tables as
;; HTML.  Inside a nested item snapper reads the lines of an HTML table
;; as prose and joins them, and so does a prose checker that only skips
;; fenced code and pipe tables.  Source, example and fixed-width blocks
;; are therefore written between backtick fences, and Org tables as
;; GitHub Flavored Markdown pipe tables.

(require 'ox-md)
(require 'seq)
(require 'subr-x)

(defun org-to-md--item (item backend _info)
  "Close up the bullet padding of ITEM, a list item ox-md transcoded.
A headline that does not start with a bullet is returned unchanged."
  (if (and (org-export-derived-backend-p backend 'md)
           (string-match "\\`\\(-\\|[0-9]+\\.\\)\\( +\\)" item))
      (let* ((bullet (match-string 1 item))
             (shift (1- (length (match-string 2 item))))
             (prefix (make-string shift ?\s))
             (lines (split-string (substring item (match-end 0)) "\n")))
        (concat bullet " " (car lines)
                (mapconcat (lambda (line)
                             (concat "\n" (if (string-prefix-p prefix line)
                                              (substring line shift)
                                            line)))
                           (cdr lines) "")))
    item))

(defun org-to-md--toc (toc)
  "Give the unnumbered entries of TOC one space after the bullet and two columns per level."
  (replace-regexp-in-string
   "^\\( *\\)-   "
   (lambda (match)
     (concat (make-string (/ (length (match-string 1 match)) 2) ?\s) "- "))
   toc t t))

(defun org-to-md--fence (code)
  "Return a fence one backtick longer than any fence-like run at a line start in CODE."
  (let ((longest 2))
    (dolist (line (split-string code "\n"))
      (when (string-match "\\` *\\(`+\\)" line)
        (setq longest (max longest (length (match-string 1 line))))))
    (make-string (1+ longest) ?`)))

(defun org-to-md--code-block (block _contents info)
  "Write BLOCK, a source, example or fixed-width block, between backtick fences."
  (let* ((code (org-remove-indentation (org-export-format-code-default block info)))
         (fence (org-to-md--fence code)))
    (concat fence "\n" code
            (unless (or (string-empty-p code) (string-suffix-p "\n" code)) "\n")
            fence)))

(defun org-to-md--cells (row table _info)
  "Return the cells of ROW, leaving out the special column of TABLE."
  (let ((cells (org-element-contents row)))
    (if (org-export-table-has-special-column-p table) (cdr cells) cells)))

(defun org-to-md--cell (cell info)
  "Transcode the contents of CELL for a pipe table, escaping its bars."
  (replace-regexp-in-string
   "|" "\\|" (org-trim (org-export-data (org-element-contents cell) info)) t t))

(defun org-to-md--pipe-table (table info)
  "Write TABLE, an Org table, as a pipe table.
The first row is the header when the table has one, and an empty
header row stands in otherwise.  Each column keeps the alignment
the HTML export would give it."
  (let* ((rows (seq-filter (lambda (row)
                             (and (eq (org-element-property :type row) 'standard)
                                  (not (org-export-table-row-is-special-p row info))))
                           (org-element-contents table)))
         (first (org-to-md--cells (car rows) table info))
         (header-p (org-export-table-has-header-p table info))
         (line (lambda (cells) (concat "| " (string-join cells " | ") " |"))))
    (mapconcat line
               (append
                (list (if header-p
                          (mapcar (lambda (cell) (org-to-md--cell cell info)) first)
                        (make-list (length first) ""))
                      (mapcar (lambda (cell)
                                (pcase (org-export-table-cell-alignment cell info)
                                  ('right "--:")
                                  ('center ":-:")
                                  (_ "---")))
                              first))
                (mapcar (lambda (row)
                          (mapcar (lambda (cell) (org-to-md--cell cell info))
                                  (org-to-md--cells row table info)))
                        (if header-p (cdr rows) rows)))
               "\n")))

(defun org-to-md--table (convert datum contents info)
  "Write DATUM as a pipe table when it is an Org table, or call CONVERT."
  (if (and (eq (org-element-type datum) 'table)
           (eq (org-element-property :type datum) 'org))
      (org-to-md--pipe-table datum info)
    (funcall convert datum contents info)))

(add-to-list 'org-export-filter-item-functions #'org-to-md--item)
(add-to-list 'org-export-filter-headline-functions #'org-to-md--item)
(advice-add 'org-md--build-toc :filter-return #'org-to-md--toc)
(advice-add 'org-md-example-block :override #'org-to-md--code-block)
(advice-add 'org-md--convert-to-html :around #'org-to-md--table)

;;; org_to_md.el ends here
