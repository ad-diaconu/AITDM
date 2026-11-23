#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Which fep partitions may I use, and with which Slurm account?
#
#   bash Fep/my_partitions.sh               # you
#   bash Fep/my_partitions.sh <username>    # someone else, e.g. a student
#
# Slurm decides by ACCOUNT (not by your role or Unix group): every partition
# lists the accounts it allows (AllowAccounts) or refuses (DenyAccounts).
# This compares those lists with your accounts. Use the account with -A, e.g.
#   bash Fep/submit.sh -A <account> -p <partition> --gres=gpu:1 simulation ...
# (`sbatch --test-only` is no help: it accepts combinations Slurm refuses later.)
# ---------------------------------------------------------------------------
set -u   # (no -e: grep finds nothing for partitions without a rule)
WHO="${1:-$USER}"
accts=$(sacctmgr -n -P show assoc user="$WHO" format=account | sort -u)
if [ -z "$accts" ]; then echo "No Slurm account found for user $WHO" >&2; exit 1; fi
groups=" $(id -Gn "$WHO" 2>/dev/null) "
echo "Accounts of $WHO: $(echo $accts) (default: $(sacctmgr -n -P show user "$WHO" format=defaultaccount))"
printf "%-12s %-10s %s\n" PARTITION STATE "ACCOUNTS YOU CAN USE"
scontrol -o show partition | while read -r line; do
  p=$(grep -oP 'PartitionName=\K\S+' <<<"$line")
  allowA=$(grep -oP ' AllowAccounts=\K\S+' <<<"$line"); denyA=$(grep -oP ' DenyAccounts=\K\S+' <<<"$line")
  allowG=$(grep -oP ' AllowGroups=\K\S+' <<<"$line")
  ok=""
  for a in $accts; do
    if [ -n "$denyA" ] && [[ ",$denyA," == *",$a,"* ]]; then continue; fi
    if [ -z "$allowA" ] || [ "$allowA" = ALL ] || [[ ",$allowA," == *",$a,"* ]]; then ok+="$a "; fi
  done
  if [ "$allowG" != ALL ] && ! tr ',' '\n' <<<"$allowG" | grep -qxF -f <(tr ' ' '\n' <<<"$groups"); then ok=""; fi
  printf "%-12s %-10s %s\n" "$p" "$(sinfo -h -p "$p" -o %a | head -1)" "${ok:-(no access)}"
done
