#!/bin/bash

EXECUTABLE=./matriz_hibrida
OUTPUT=resultados.csv

echo "Procesos,Threads,Tiempo(s)" > $OUTPUT

for procesos in 1 2 4 8
do
    for hilos in 1 2 4
    do
        export OMP_NUM_THREADS=$hilos
        echo "Ejecutando con $procesos procesos y $hilos hilos por proceso..."

        salida=$(mpirun --oversubscribe -np $procesos $EXECUTABLE 2>&1)
        linea=$(echo "$salida" | grep "Tiempo total")

        if [[ -n "$linea" ]]; then
            tiempo=$(echo "$linea" | awk '{print $(NF-1)}')
            echo "$procesos,$hilos,$tiempo" >> $OUTPUT
        else
            echo "$procesos,$hilos,ERROR" >> $OUTPUT
            echo "[!] Error ejecutando con $procesos procesos y $hilos hilos."
        fi
    done
done

echo "Pruebas completadas. Revisa $OUTPUT"
