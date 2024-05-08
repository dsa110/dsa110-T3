# GReX-T3
The T3 subsystem is meant to process single-pulse candidates after detection and clustering. T3 will coincidence with other GReX terminals within a cluster, classify dynamic spectra with an ML model, and plot events.

# Installation 

We have been using "Poetry" for packaging and dependency management. To install using Poetry, enter the GReX-T3 dir and do 

$ poetry install 

Edit "pyproject.toml" to manage relevant Python packages, or do "poetry add <package>". 

# Services
T3 will have several background system services that will watch over filesystems, process triggers, etc. Typical usage is as follows,

service <service_name> start|stop|restart|status

For example, if we want to check on the service that clears voltage triggers and ensures there's enough disk space, we can do 

sudo systemctl status clear_disks.service

and/or,

journalctl -u clear_disks.service
