#!/bin/bash
for host in hopper lovelace johnson; do
  rsync -au $host:value-alignment-verification/data/ /home/joschnei/value-alignment-verification/data/
done
for host in hopper lovelace johnson; do
  rsync -au /home/joschnei/value-alignment-verification/data/ $host:value-alignment-verification/data 
done
