#!/bin/bash

# Definisi Variabel
DB_HOST="10.213.224.113"
DB_PORT="5432"
DB_USERNAME="bribrain_user"
DB_NAME="bribrain_knowledge_base"
# Gunakan kutip satu (' ') untuk password yang ada spesial karakternya
DB_PASSWORD='Bribrainaj4!' 

# Eksekusi Command
# export PGPASSWORD agar bisa dibaca oleh psql tanpa prompt
export PGPASSWORD="$DB_PASSWORD"

psql -h "$DB_HOST" \
     -p "$DB_PORT" \
     -U "$DB_USERNAME" \
     -d "$DB_NAME" \
     -f scripts/embedding_migration/alter_dimension.sql

# Opsional: Hapus variable password dari environment setelah selesai demi keamanan
unset PGPASSWORD