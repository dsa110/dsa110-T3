#! /usr/bin/env bash

systemctl --user stop send_cands.service
systemctl --user stop dask_scheduler.service
systemctl --user stop dask_worker.service

sleep 2
systemctl --user start dask_scheduler.service
systemctl --user start dask_worker.service
systemctl --user start send_cands.service



