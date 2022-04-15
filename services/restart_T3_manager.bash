#! /usr/bin/env bash

systemctl --user stop send_cands.service
systemctl --user stop dask_scheduler.service
systemctl --user stop dask_worker.service
systemctl --user stop dask_worker2.service
systemctl --user stop dask_worker3.service
systemctl --user stop dask_worker4.service
systemctl --user stop dask_worker5.service
systemctl --user stop dask_worker6.service
systemctl --user stop dask_worker7.service
systemctl --user stop dask_worker8.service

sleep 2

systemctl --user start dask_scheduler.service
systemctl --user start dask_worker.service
#systemctl --user start dask_worker2.service
#systemctl --user start dask_worker3.service
#systemctl --user start dask_worker4.service
#systemctl --user start dask_worker5.service
#systemctl --user start dask_worker6.service
#systemctl --user start dask_worker7.service
#systemctl --user start dask_worker8.service
systemctl --user start send_cands.service



