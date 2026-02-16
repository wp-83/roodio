<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('songs', function (Blueprint $table) {
            $table->char('id', 10)->primary();
            $table->char('userId', 10);
            $table->foreign('userId')
                ->references('id')
                ->on('users')
                ->noActionOnDelete()
                ->noActionOnUpdate();
            $table->char('moodId', 10);
            $table->foreign('moodId')
                ->references('id')
                ->on('moods')
                ->noActionOnDelete()
                ->noActionOnUpdate();
            $table->integer('confidence');
            $table->string('title', 255);
            $table->text('lyrics');
            $table->string('artist', 255);
            $table->string('genre', 255);
            $table->integer('duration');
            $table->string('publisher', 255);
            $table->date('datePublished');
            $table->string('songPath', 255)->nullable();
            $table->string('photoPath', 255)->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('songs');
    }
};
