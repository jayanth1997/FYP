package tweetpre;

import java.io.FileNotFoundException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;
 
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

/**
 *
 * @author user12345
 */
public class Tweetpre {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
         String URL_REGEX = "((www\\.[\\s]+)|(https?://[^\\s]+))"; 

 String STARTS_WITH_NUMBER = "[0-9]\\s*(\\w+)";
 String STARTS_WITH_NUMBER1 = "[#][a-zA-Z0-9]\\s*(\\w+)";
 String u = "[ud][a-zA-Z0-9]\\s*(\\w+)";
 String num="[a-zA-z][0-9]";
                        XSSFWorkbook workbook1 = new XSSFWorkbook();
                        XSSFSheet sheet1 = workbook1.createSheet("Sample sheet");
        String excelFilePath = "C:\\python\\without hashtags\\preprocessed\\Book2.xlsx";
        try (FileInputStream inputStream = new FileInputStream(new File(excelFilePath))) {
            String tweet;
         
            int rownum = 0;
             try (Workbook workbook = new XSSFWorkbook(inputStream)) {
                 Sheet firstSheet = workbook.getSheetAt(0);
                 Iterator<Row> iterator = firstSheet.iterator();
                 
                 while (iterator.hasNext()) {
                     Row nextRow = iterator.next();
                     Iterator<Cell> cellIterator = nextRow.cellIterator();
                     
                     while (cellIterator.hasNext()) {
                         Cell cell = cellIterator.next();
                         tweet= cell.getStringCellValue();
                         tweet = tweet.replaceAll(URL_REGEX, "");
                         
                         // Remove @username
                         tweet = tweet.replaceAll("@([^\\s]+)", "");
                         
//tweet = tweet.replaceAll("\\<.*?\\>", "");
                         // Remove character repetition
                         tweet = tweet.replaceAll("RT", "");
                         tweet = tweet.replaceAll("u", "");
                         // Remove words starting with a number
                         tweet = tweet.replaceAll(STARTS_WITH_NUMBER, "");
                         
                         
                         tweet = tweet.replaceAll(STARTS_WITH_NUMBER1, "");
                         // tweet = tweet.replaceAll(ud, "");
                         
                         tweet = tweet.replaceAll("[^a-zA-Z0-9 ]+", "").trim();
                         tweet = tweet.replaceAll("num", "").trim();
                         
                        //-------------------------------------------------------------------------------------
                         
                        


 //---------------------------------------------------------------------------------------------
                        Row row = sheet1.createRow(rownum++);
			int cellnum = 0;
                        Cell cell1 = row.createCell(cellnum);
			cell1.setCellValue((String)tweet);
				
			}
		}
		try {
                     try (FileOutputStream out = new FileOutputStream(new File("C:\\python\\without hashtags\\preprocessed\\Book2p1.csv"))) {
                         workbook1.write(out);
                     }
			System.out.println("Excel written successfully..");
			
		} catch (FileNotFoundException e) {
		} catch (IOException e) {
		}
                         
                     }
                     
                 }
             }
        }
 